package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Hand Landmark Detector using TFLite with NNAPI/GPU acceleration
 *
 * Detects 21 hand landmarks from ROI extracted by HandDetector
 * Uses MediaPipe hand landmark model
 *
 * Performance with delegates:
 *   - NNAPI (NPU): 2-8ms
 *   - GPU: 3-12ms
 *   - CPU: 25-50ms
 */
class HandLandmarkDetectorTFLite(private val context: Context) {

    companion object {
        private const val TAG = "HandLandmarkDetectorTFLite"
        private const val MODEL_NAME = "mediapipe_hand-handlandmarkdetector.tflite"

        // Model constants
        private const val INPUT_SIZE = 256
        private const val NUM_LANDMARKS = 21

        // Thresholds
        private const val PRESENCE_THRESH = 0.5f
    }

    private var interpreter: Interpreter? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var gpuDelegate: GpuDelegate? = null
    private var actualBackend = "UNKNOWN"

    // Input/output buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputScores: FloatArray  // Changed from Array<FloatArray>
    private lateinit var outputHandedness: FloatArray  // Changed from Array<FloatArray>
    private lateinit var outputLandmarks: Array<Array<FloatArray>>

    init {
        Log.d(TAG, "════════════════════════════════════════")
        Log.d(TAG, "Initializing Landmark Detector (TFLite)")
        FileLogger.section("Initializing Landmark Detector (TFLite)")

        // Load model with delegates
        loadModel()

        // Allocate buffers
        allocateBuffers()

        Log.d(TAG, "✓ Landmark Detector ready on $actualBackend")
        Log.d(TAG, "════════════════════════════════════════")
        FileLogger.i(TAG, "✓ Landmark Detector ready on $actualBackend")
    }

    /**
     * Load TFLite model with multi-tier delegate fallback
     */
    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile(context, MODEL_NAME)
            val options = Interpreter.Options()

            // ═══════════════════════════════════════════════════════
            // TIER 1: Try NNAPI (NPU preferred)
            // ═══════════════════════════════════════════════════════
            try {
                nnApiDelegate = NnApiDelegate(
                    NnApiDelegate.Options().apply {
                        setExecutionPreference(
                            NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER
                        )
                        setAllowFp16(true)
                    }
                )
                options.addDelegate(nnApiDelegate)
                actualBackend = "NNAPI (NPU preferred)"
                Log.d(TAG, "✓ Using NNAPI delegate (NPU preferred)")

            } catch (e: Exception) {
                Log.w(TAG, "NNAPI delegate failed: ${e.message}")

                // ═══════════════════════════════════════════════════════
                // TIER 2: Try GPU Delegate
                // ═══════════════════════════════════════════════════════
                try {
                    gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                    actualBackend = "GPU (OpenGL)"
                    Log.d(TAG, "✓ Using GPU delegate (fallback)")

                } catch (e2: Exception) {
                    Log.w(TAG, "GPU delegate failed: ${e2.message}")

                    // ═══════════════════════════════════════════════════════
                    // TIER 3: CPU with threading
                    // ═══════════════════════════════════════════════════════
                    options.setNumThreads(4)
                    actualBackend = "CPU (4 threads)"
                    Log.d(TAG, "✓ Using CPU with 4 threads (fallback)")
                }
            }

            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "✓ Model loaded: ${modelBuffer.capacity() / 1024}KB")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            throw e
        }
    }

    /**
     * Allocate input/output buffers
     */
    private fun allocateBuffers() {
        // Input: [1, 256, 256, 3] float32
        inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        // FIXED: Output 0: scores [1] - single float value
        outputScores = FloatArray(1)

        // FIXED: Output 1: handedness [1] - single float value
        outputHandedness = FloatArray(1)

        // Output 2: landmarks [1, 21, 3] (x, y, z)
        outputLandmarks = Array(1) { Array(NUM_LANDMARKS) { FloatArray(3) } }
    }

    /**
     * Detect hand landmarks from ROI
     *
     * @param frame Original frame
     * @param roi Region of interest from detector
     * @return LandmarkResult or null if presence too low
     */
    fun detectLandmarks(frame: Bitmap, roi: HandROI): LandmarkResult? {
        FileLogger.d(TAG, "detectLandmarks() called")

        // Check if interpreter is initialized
        if (interpreter == null) {
            FileLogger.e(TAG, "❌ Interpreter is NULL! Model failed to load!")
            return null
        }

        try {
            // Warp ROI to 256×256 square (affine transform)
            FileLogger.d(TAG, "Warping ROI to 256x256...")
            val warped = warpROI(frame, roi)
            FileLogger.d(TAG, "✓ ROI warped")

            // Preprocess
            FileLogger.d(TAG, "Preprocessing...")
            preprocessImage(warped)
            FileLogger.d(TAG, "✓ Preprocessing done")

            // Run inference
            FileLogger.d(TAG, "Running inference...")
            val outputs = mapOf(
                0 to outputScores,
                1 to outputHandedness,
                2 to outputLandmarks
            )
            interpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
            FileLogger.d(TAG, "✓ Inference completed")

            // Check presence - FIXED: access as FloatArray[0] not [0][0]
            val presence = outputScores[0]
            FileLogger.d(TAG, "Presence score: $presence (threshold: $PRESENCE_THRESH)")

            if (presence < PRESENCE_THRESH) {
                FileLogger.d(TAG, "Hand presence too low, returning null")
                return null
            }

            // Unproject landmarks back to frame coordinates
            FileLogger.d(TAG, "Unprojecting landmarks to frame coordinates...")
            val landmarksNormalized = outputLandmarks[0]  // [21, 3] in [0, 1]
            val landmarksFrame = unprojectLandmarks(landmarksNormalized, roi)

            // Determine handedness - FIXED: access as FloatArray[0] not [0][0]
            val handedness = if (outputHandedness[0] > 0.5f) "Right" else "Left"
            FileLogger.d(TAG, "✓ Landmarks detected! Handedness: $handedness, Presence: $presence")

            return LandmarkResult(
                landmarks = landmarksFrame,
                presence = presence,
                handedness = handedness
            )

        } catch (e: Exception) {
            Log.e(TAG, "Landmark detection failed", e)
            FileLogger.e(TAG, "Landmark detection failed!", e)
            return null
        }
    }

    /**
     * Warp ROI region to 256×256 square using affine transform
     */
    private fun warpROI(frame: Bitmap, roi: HandROI): Bitmap {
        // Source points: top-left, top-right, bottom-right corners of ROI
        val srcPoints = floatArrayOf(
            roi.rectPoints[1][0], roi.rectPoints[1][1],  // top-left
            roi.rectPoints[2][0], roi.rectPoints[2][1],  // top-right
            roi.rectPoints[3][0], roi.rectPoints[3][1]   // bottom-right
        )

        // Destination points: 256×256 square
        val dstPoints = floatArrayOf(
            0f, 0f,                    // top-left
            INPUT_SIZE.toFloat(), 0f,  // top-right
            INPUT_SIZE.toFloat(), INPUT_SIZE.toFloat()  // bottom-right
        )

        // Calculate affine transformation matrix
        val matrix = Matrix()
        matrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 3)

        // Create warped bitmap
        val warped = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(warped)
        canvas.drawBitmap(frame, matrix, null)

        return warped
    }

    /**
     * Preprocess image for landmark detector
     * Input: RGB Bitmap (256×256)
     * Output: RGB normalized to [0, 1] in NHWC format
     */
    private fun preprocessImage(bitmap: Bitmap) {
        inputBuffer.rewind()

        // Convert to RGB, normalize to [0, 1]
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            // Extract RGB (bitmap is ARGB)
            val r = ((pixel shr 16) and 0xFF)
            val g = ((pixel shr 8) and 0xFF)
            val b = (pixel and 0xFF)

            // Normalize: pixel / 255.0 → [0, 1]
            inputBuffer.putFloat(r / 255.0f)
            inputBuffer.putFloat(g / 255.0f)
            inputBuffer.putFloat(b / 255.0f)
        }
    }

    /**
     * Unproject landmarks from normalized [0,1] back to frame coordinates
     */
    private fun unprojectLandmarks(
        landmarksNorm: Array<FloatArray>,
        roi: HandROI
    ): Array<FloatArray> {

        // Source points: normalized [0, 1] space
        val srcPoints = floatArrayOf(
            0f, 0f,     // top-left
            1f, 0f,     // top-right
            1f, 1f      // bottom-right
        )

        // Destination points: ROI corners in frame coordinates
        val dstPoints = floatArrayOf(
            roi.rectPoints[1][0], roi.rectPoints[1][1],  // top-left
            roi.rectPoints[2][0], roi.rectPoints[2][1],  // top-right
            roi.rectPoints[3][0], roi.rectPoints[3][1]   // bottom-right
        )

        // Calculate inverse affine transformation
        val matrix = Matrix()
        matrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 3)

        // Transform all landmarks
        val landmarksFrame = Array(NUM_LANDMARKS) { FloatArray(3) }
        val pointsIn = FloatArray(2)
        val pointsOut = FloatArray(2)

        for (i in 0 until NUM_LANDMARKS) {
            pointsIn[0] = landmarksNorm[i][0]
            pointsIn[1] = landmarksNorm[i][1]

            matrix.mapPoints(pointsOut, pointsIn)

            landmarksFrame[i][0] = pointsOut[0]  // x in frame coords
            landmarksFrame[i][1] = pointsOut[1]  // y in frame coords
            landmarksFrame[i][2] = landmarksNorm[i][2]  // z stays normalized
        }

        return landmarksFrame
    }

    /**
     * Load TFLite model file from assets
     */
    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Get actual backend being used
     */
    fun getBackend(): String = actualBackend

    /**
     * Release resources
     */
    fun close() {
        interpreter?.close()
        nnApiDelegate?.close()
        gpuDelegate?.close()
        Log.d(TAG, "✓ Landmark Detector closed")
    }
}

/**
 * Landmark detection result
 */
data class LandmarkResult(
    val landmarks: Array<FloatArray>,  // [21, 3] - x, y, z in frame coords
    val presence: Float,
    val handedness: String  // "Left" or "Right"
)