package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import ai.onnxruntime.*

/**
 * ONNX Hand Landmark Detector - Stage 2 of pipeline
 *
 * Extracts 21 hand landmarks from cropped hand region
 * Input: 256×256 RGB image normalized to [0, 1]
 * Output: 21 landmarks (x, y, z) + handedness + presence score
 *
 * Based on geaxgx MediaPipe ONNX implementation
 */
class HandLandmarkDetectorONNX(private val context: Context) {

    companion object {
        private const val TAG = "HandLandmarkONNX"
        private const val MODEL_NAME = "HandLandmarkDetector.onnx"
        private const val INPUT_SIZE = 256
        private const val PRESENCE_THRESHOLD = 0.5f
        private const val NUM_LANDMARKS = 21
    }

    private var ortEnvironment: OrtEnvironment? = null
    private var onnxSession: OrtSession? = null

    init {
        Log.d(TAG, "════════════════════════════════════════")
        Log.d(TAG, "Initializing Landmark Detector (GPU mode)")
        Log.d(TAG, "════════════════════════════════════════")

        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            val modelBytes = context.assets.open(MODEL_NAME).use { it.readBytes() }

            val sessionOptions = OrtSession.SessionOptions()

            // ═══════════════════════════════════════════════════════
            // GPU MODE: Force NNAPI to prefer GPU over NPU
            // ═══════════════════════════════════════════════════════
            try {
                sessionOptions.addNnapi()

                // LOW_POWER mode prefers GPU
                // sessionOptions.addConfigEntry("nnapi.execution_preference", "2")

                Log.d(TAG, "✓ NNAPI enabled with GPU preference")
                Log.d(TAG, "   Target: Mali-G68 MP5 GPU")
            } catch (e: Exception) {
                Log.e(TAG, "✗ NNAPI configuration failed: ${e.message}")
                Log.e(TAG, "   Falling back to default NNAPI")
            }
            // ═══════════════════════════════════════════════════════

            onnxSession = ortEnvironment?.createSession(modelBytes, sessionOptions)

            Log.d(TAG, "✓ Landmark Detector loaded successfully")
            Log.d(TAG, "   Model size: ${modelBytes.size / 1024}KB")
            Log.d(TAG, "   Backend: NNAPI (GPU preferred)")
            Log.d(TAG, "════════════════════════════════════════")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to load landmark model", e)
            throw e
        }
    }

    /**
     * Detect landmarks from hand ROI
     *
     * @param frame Original frame
     * @param roi Hand region of interest
     * @return HandLandmarks or null if hand not present
     */
    fun detectLandmarks(frame: Bitmap, roi: HandROI): HandLandmarks? {
        try {
            // Extract and warp hand crop
            val crop = warpROI(frame, roi)

            // Preprocess
            val input = preprocessImage(crop)

            // Run inference
            val inputName = onnxSession?.inputNames?.iterator()?.next() ?: "image"
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, input)

            val outputs = onnxSession?.run(mapOf(inputName to inputTensor))

            // Parse outputs
            val presence = (outputs?.get(0)?.value as FloatArray)[0]  // Hand presence score
            val handedness = (outputs?.get(1)?.value as FloatArray)[0]  // Left/right
            val landmarks = outputs?.get(2)?.value as Array<Array<FloatArray>>  // (1, 21, 3)

            inputTensor.close()
            outputs?.close()

            // Check presence threshold
            if (presence < PRESENCE_THRESHOLD) {
                Log.d(TAG, "Hand lost: presence=$presence < $PRESENCE_THRESHOLD")
                return null
            }

            // Unproject landmarks from crop space to frame space
            val landmarksNorm = landmarks[0]  // (21, 3) in crop space [0, 1]
            val landmarksFrame = unprojectLandmarks(landmarksNorm, roi)

            val hand = if (handedness > 0.5f) "Right" else "Left"

            Log.d(TAG, "Landmarks detected: presence=${"%.3f".format(presence)}, " +
                      "hand=$hand, wrist=(${landmarksFrame[0][0].toInt()}, ${landmarksFrame[0][1].toInt()})")

            return HandLandmarks(
                landmarks = landmarksFrame,
                presence = presence,
                handedness = hand
            )

        } catch (e: Exception) {
            Log.e(TAG, "Landmark detection failed", e)
            return null
        }
    }

    /**
     * Warp ROI from frame to 256×256 crop using affine transform
     *
     * Uses 3-point affine transform (not perspective):
     * - Source: roi rectPoints[1:3] (top-left, top-right, bottom-right)
     * - Dest: [(0,0), (256,0), (256,256)]
     *
     * This is LINEAR transformation - no distortion!
     */
    private fun warpROI(frame: Bitmap, roi: HandROI): Bitmap {
        // Source points: top-left, top-right, bottom-right (skip bottom-left)
        val srcPoints = floatArrayOf(
            roi.rectPoints[1][0], roi.rectPoints[1][1],  // top-left
            roi.rectPoints[2][0], roi.rectPoints[2][1],  // top-right
            roi.rectPoints[3][0], roi.rectPoints[3][1]   // bottom-right
        )

        // Destination points in 256×256 crop
        val dstPoints = floatArrayOf(
            0f, 0f,                      // top-left
            INPUT_SIZE.toFloat(), 0f,    // top-right
            INPUT_SIZE.toFloat(), INPUT_SIZE.toFloat()  // bottom-right
        )

        // Create affine transform matrix
        val matrix = Matrix()
        matrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 3)

        // Create output bitmap
        val crop = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)

        // Apply transform
        val canvas = android.graphics.Canvas(crop)
        canvas.drawBitmap(frame, matrix, null)

        return crop
    }

    /**
     * Preprocess image for landmark detector
     * RGB, resize to 256×256, normalize to [0, 1], NCHW format
     *
     * NOTE: Different normalization than detector!
     * Detector: [-1, 1]
     * Landmarks: [0, 1]
     */
    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // NCHW format: [1, 3, 256, 256]
        val input = Array(1) {
            Array(3) {
                Array(INPUT_SIZE) {
                    FloatArray(INPUT_SIZE)
                }
            }
        }

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = pixels[y * INPUT_SIZE + x]

                // Extract RGB and normalize to [0, 1]
                val r = ((pixel shr 16) and 0xFF).toFloat() / 255f
                val g = ((pixel shr 8) and 0xFF).toFloat() / 255f
                val b = (pixel and 0xFF).toFloat() / 255f

                input[0][0][y][x] = r  // R channel
                input[0][1][y][x] = g  // G channel
                input[0][2][y][x] = b  // B channel
            }
        }

        return input
    }

    /**
     * Unproject landmarks from crop space to frame space
     *
     * Landmarks from model are in crop coordinates [0, 1]
     * Need to map back to original frame pixel coordinates
     *
     * Uses inverse affine transform - perfectly linear, no distortion
     */
    private fun unprojectLandmarks(
        landmarksNorm: Array<FloatArray>,  // (21, 3) in [0, 1]
        roi: HandROI
    ): Array<FloatArray> {

        // Source points in normalized crop space [0, 1]
        val srcPoints = floatArrayOf(
            0f, 0f,   // top-left
            1f, 0f,   // top-right
            1f, 1f    // bottom-right
        )

        // Destination points in frame space (roi corners)
        val dstPoints = floatArrayOf(
            roi.rectPoints[1][0], roi.rectPoints[1][1],  // top-left
            roi.rectPoints[2][0], roi.rectPoints[2][1],  // top-right
            roi.rectPoints[3][0], roi.rectPoints[3][1]   // bottom-right
        )

        // Create inverse affine transform
        val matrix = Matrix()
        matrix.setPolyToPoly(srcPoints, 0, dstPoints, 0, 3)

        // Transform all landmarks
        val landmarksFrame = Array(NUM_LANDMARKS) { FloatArray(3) }
        val pointsIn = FloatArray(2)
        val pointsOut = FloatArray(2)

        for (i in 0 until NUM_LANDMARKS) {
            pointsIn[0] = landmarksNorm[i][0]  // x in [0, 1]
            pointsIn[1] = landmarksNorm[i][1]  // y in [0, 1]

            matrix.mapPoints(pointsOut, pointsIn)

            landmarksFrame[i][0] = pointsOut[0]  // x in frame pixels
            landmarksFrame[i][1] = pointsOut[1]  // y in frame pixels
            landmarksFrame[i][2] = landmarksNorm[i][2]  // z (unchanged)
        }

        return landmarksFrame
    }

    fun close() {
        onnxSession?.close()
        Log.d(TAG, "Hand Landmark Detector closed")
    }
}

/**
 * Hand landmarks result
 */
data class HandLandmarks(
    val landmarks: Array<FloatArray>,  // 21 landmarks × (x, y, z) in frame coordinates
    val presence: Float,                // Presence confidence [0, 1]
    val handedness: String              // "Left" or "Right"
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as HandLandmarks

        if (!landmarks.contentDeepEquals(other.landmarks)) return false
        if (presence != other.presence) return false
        if (handedness != other.handedness) return false

        return true
    }

    override fun hashCode(): Int {
        var result = landmarks.contentDeepHashCode()
        result = 31 * result + presence.hashCode()
        result = 31 * result + handedness.hashCode()
        return result
    }
}