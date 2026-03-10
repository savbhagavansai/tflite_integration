package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.*

/**
 * Hand Detector using TFLite with NNAPI/GPU acceleration
 *
 * Detects hands in full frame and returns ROI for landmark detection
 * Uses MediaPipe hand detector model (SSD with 2944 anchors)
 *
 * Performance with delegates:
 *   - NNAPI (NPU): 3-10ms
 *   - GPU: 5-15ms
 *   - CPU: 30-60ms
 */
class HandDetectorTFLite(private val context: Context) {

    companion object {
        private const val TAG = "HandDetectorTFLite"
        private const val MODEL_NAME = "mediapipe_hand-handdetector.tflite"

        // Model constants
        private const val INPUT_SIZE = 256
        private const val NUM_ANCHORS = 2944
        private const val DECODE_SCALE = 256f

        // Detection thresholds
        private const val DETECTOR_THRESH = 0.5f
        private const val NMS_IOU_THRESH = 0.3f

        // ROI transformation constants
        private const val SCALE_X = 2.9f
        private const val SCALE_Y = 2.9f
        private const val SHIFT_X = 0.0f
        private const val SHIFT_Y = -0.5f
    }

    private var interpreter: Interpreter? = null
    private var nnApiDelegate: NnApiDelegate? = null
    private var gpuDelegate: GpuDelegate? = null
    private var actualBackend = "UNKNOWN"

    // Anchors for box decoding (generated once)
    private val anchors: FloatArray

    // Input/output buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBoxes: Array<Array<FloatArray>>
    private lateinit var outputScores: Array<Array<FloatArray>>

    init {
        Log.d(TAG, "════════════════════════════════════════")
        Log.d(TAG, "Initializing Hand Detector (TFLite)")

        // Generate anchors
        anchors = generateAnchors()
        Log.d(TAG, "✓ Generated ${anchors.size / 4} anchors")

        // Load model with delegates
        loadModel()

        // Allocate buffers
        allocateBuffers()

        Log.d(TAG, "✓ Hand Detector ready on $actualBackend")
        Log.d(TAG, "════════════════════════════════════════")
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
                        // FAST_SINGLE_ANSWER = prefer NPU over GPU
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
     * Generate anchors for box decoding
     * Spec: [(stride=8, count=2), (stride=16, count=2), (stride=32, count=6)]
     */
    private fun generateAnchors(): FloatArray {
        val spec = arrayOf(
            Pair(8, 2),   // stride 8: 32×32 grid × 2 = 2048 anchors
            Pair(16, 2),  // stride 16: 16×16 grid × 2 = 512 anchors
            Pair(32, 6)   // stride 32: 8×8 grid × 6 = 384 anchors
        )

        val anchorList = mutableListOf<Float>()

        for ((stride, count) in spec) {
            val grid = INPUT_SIZE / stride
            for (y in 0 until grid) {
                for (x in 0 until grid) {
                    val cx = (x + 0.5f) / grid
                    val cy = (y + 0.5f) / grid
                    repeat(count) {
                        anchorList.add(cx)
                        anchorList.add(cy)
                        anchorList.add(1.0f)  // width
                        anchorList.add(1.0f)  // height
                    }
                }
            }
        }

        return anchorList.toFloatArray()
    }

    /**
     * Allocate input/output buffers
     */
    private fun allocateBuffers() {
        // Input: [1, 256, 256, 3] float32
        inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Output 0: raw_boxes [1, 2944, 18]
        outputBoxes = Array(1) { Array(NUM_ANCHORS) { FloatArray(18) } }

        // Output 1: raw_scores [1, 2944, 1]
        outputScores = Array(1) { Array(NUM_ANCHORS) { FloatArray(1) } }
    }

    /**
     * Detect hand in frame
     *
     * @param bitmap Input frame (any size)
     * @return HandDetection or null if no hand detected
     */
    fun detectHand(bitmap: Bitmap): HandDetection? {
        try {
            // Preprocess
            preprocessImage(bitmap)

            // Run inference
            val outputs = mapOf(
                0 to outputBoxes,
                1 to outputScores
            )
            interpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

            // Process detections
            return processDetections(bitmap.width, bitmap.height)

        } catch (e: Exception) {
            Log.e(TAG, "Detection failed", e)
            return null
        }
    }

    /**
     * Preprocess image for detector
     * Input: BGR Bitmap
     * Output: RGB normalized to [-1, 1] in NHWC format
     */
    private fun preprocessImage(bitmap: Bitmap) {
        // Resize to 256×256
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        inputBuffer.rewind()

        // Convert to RGB, normalize to [-1, 1]
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            // Extract RGB (bitmap is ARGB)
            val r = ((pixel shr 16) and 0xFF)
            val g = ((pixel shr 8) and 0xFF)
            val b = (pixel and 0xFF)

            // Normalize: (pixel - 127.5) / 127.5 → [-1, 1]
            inputBuffer.putFloat((r - 127.5f) / 127.5f)
            inputBuffer.putFloat((g - 127.5f) / 127.5f)
            inputBuffer.putFloat((b - 127.5f) / 127.5f)
        }
    }

    /**
     * Process raw model outputs to detect hand
     */
    private fun processDetections(frameWidth: Int, frameHeight: Int): HandDetection? {
        val rawBoxes = outputBoxes[0]
        val rawScores = outputScores[0]

        // Apply sigmoid to scores
        val scores = FloatArray(NUM_ANCHORS)
        for (i in scores.indices) {
            scores[i] = sigmoid(rawScores[i][0])
        }

        // Filter by threshold
        val validIndices = mutableListOf<Int>()
        for (i in scores.indices) {
            if (scores[i] > DETECTOR_THRESH) {
                validIndices.add(i)
            }
        }

        if (validIndices.isEmpty()) return null

        // Decode boxes for valid detections
        val (boxes, keypoints) = decodeBoxes(rawBoxes, validIndices)

        // Apply NMS
        val validScores = validIndices.map { scores[it] }.toFloatArray()
        val kept = nms(boxes, validScores)

        if (kept.isEmpty()) return null

        // Take best detection
        val bestIdx = kept[0]
        val box = boxes[bestIdx]
        val kps = keypoints[bestIdx]
        val confidence = validScores[bestIdx]

        // Build ROI from detection
        val roi = buildDetectionROI(box, kps, frameWidth, frameHeight)

        return HandDetection(
            roi = roi,
            confidence = confidence,
            keypoints = kps
        )
    }

    /**
     * Decode bounding boxes from anchor-based predictions
     */
    private fun decodeBoxes(
        rawBoxes: Array<FloatArray>,
        validIndices: List<Int>
    ): Pair<Array<FloatArray>, Array<Array<FloatArray>>> {

        val boxes = Array(validIndices.size) { FloatArray(4) }
        val keypoints = Array(validIndices.size) { Array(7) { FloatArray(2) } }

        validIndices.forEachIndexed { i, anchorIdx ->
            val raw = rawBoxes[anchorIdx]  // 18 values
            val anchorBase = anchorIdx * 4

            val anchorCx = anchors[anchorBase]
            val anchorCy = anchors[anchorBase + 1]
            val anchorW = anchors[anchorBase + 2]
            val anchorH = anchors[anchorBase + 3]

            // Decode center, width, height
            val cx = raw[0] * anchorW / DECODE_SCALE + anchorCx
            val cy = raw[1] * anchorH / DECODE_SCALE + anchorCy
            val w = raw[2] * anchorW / DECODE_SCALE
            val h = raw[3] * anchorH / DECODE_SCALE

            // Convert to [x1, y1, x2, y2]
            boxes[i][0] = (cx - w / 2).coerceIn(0f, 1f)  // x1
            boxes[i][1] = (cy - h / 2).coerceIn(0f, 1f)  // y1
            boxes[i][2] = (cx + w / 2).coerceIn(0f, 1f)  // x2
            boxes[i][3] = (cy + h / 2).coerceIn(0f, 1f)  // y2

            // Decode 7 keypoints
            for (kp in 0 until 7) {
                val kpx = raw[4 + kp * 2] * anchorW / DECODE_SCALE + anchorCx
                val kpy = raw[5 + kp * 2] * anchorH / DECODE_SCALE + anchorCy
                keypoints[i][kp][0] = kpx.coerceIn(0f, 1f)
                keypoints[i][kp][1] = kpy.coerceIn(0f, 1f)
            }
        }

        return Pair(boxes, keypoints)
    }

    /**
     * Non-Maximum Suppression
     */
    private fun nms(boxes: Array<FloatArray>, scores: FloatArray): List<Int> {
        val order = scores.indices.sortedByDescending { scores[it] }
        val kept = mutableListOf<Int>()
        val suppressed = BooleanArray(boxes.size)

        for (i in order) {
            if (suppressed[i]) continue

            kept.add(i)
            if (kept.size >= 1) break  // Only need top detection

            // Suppress overlapping boxes
            for (j in order) {
                if (suppressed[j] || i == j) continue
                if (iou(boxes[i], boxes[j]) > NMS_IOU_THRESH) {
                    suppressed[j] = true
                }
            }
        }

        return kept
    }

    /**
     * Calculate IoU between two boxes
     */
    private fun iou(box1: FloatArray, box2: FloatArray): Float {
        val x1 = maxOf(box1[0], box2[0])
        val y1 = maxOf(box1[1], box2[1])
        val x2 = minOf(box1[2], box2[2])
        val y2 = minOf(box1[3], box2[3])

        val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        val area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6f)
    }

    /**
     * Build ROI from detection
     */
    private fun buildDetectionROI(
        box: FloatArray,
        keypoints: Array<FloatArray>,
        frameWidth: Int,
        frameHeight: Int
    ): HandROI {

        // Keypoints: 0=wrist, 2=middle_mcp
        val wrist = keypoints[0]
        val middleMcp = keypoints[2]

        // Calculate rotation
        val rotation = normalizeRadians(
            0.5f * PI.toFloat() - atan2(-(middleMcp[1] - wrist[1]), middleMcp[0] - wrist[0])
        )

        // Box center
        val bx = box[0]
        val by = box[1]
        val bw = box[2] - box[0]
        val bh = box[3] - box[1]
        val rectCx = bx + bw / 2
        val rectCy = by + bh / 2

        // Apply transformation
        val (cxA, cyA) = if (abs(rotation) < 1e-6) {
            Pair(
                (rectCx + bw * SHIFT_X) * frameWidth,
                (rectCy + bh * SHIFT_Y) * frameHeight
            )
        } else {
            val xShift = frameWidth * bw * SHIFT_X * cos(rotation) -
                        frameHeight * bh * SHIFT_Y * sin(rotation)
            val yShift = frameWidth * bw * SHIFT_X * sin(rotation) +
                        frameHeight * bh * SHIFT_Y * cos(rotation)
            Pair(
                rectCx * frameWidth + xShift,
                rectCy * frameHeight + yShift
            )
        }

        // Square ROI (scale by longer side)
        val longSide = maxOf(bw * frameWidth, bh * frameHeight)
        val wA = longSide * SCALE_X
        val hA = longSide * SCALE_Y

        // Calculate 4 corner points
        val rectPoints = rotatedRectToPoints(cxA, cyA, wA, hA, rotation)

        return HandROI(
            rotation = rotation,
            centerX = cxA,
            centerY = cyA,
            width = wA,
            height = hA,
            rectPoints = rectPoints,
            frameWidth = frameWidth,
            frameHeight = frameHeight
        )
    }

    /**
     * Calculate 4 corners of rotated rectangle
     */
    private fun rotatedRectToPoints(
        cx: Float, cy: Float,
        w: Float, h: Float,
        rotation: Float
    ): Array<FloatArray> {
        val b = cos(rotation) * 0.5f
        val a = sin(rotation) * 0.5f

        val p0x = cx - a * h - b * w
        val p0y = cy + b * h - a * w
        val p1x = cx + a * h - b * w
        val p1y = cy - b * h - a * w
        val p2x = 2 * cx - p0x
        val p2y = 2 * cy - p0y
        val p3x = 2 * cx - p1x
        val p3y = 2 * cy - p1y

        return arrayOf(
            floatArrayOf(p0x, p0y),  // bottom-left
            floatArrayOf(p1x, p1y),  // top-left
            floatArrayOf(p2x, p2y),  // top-right
            floatArrayOf(p3x, p3y)   // bottom-right
        )
    }

    /**
     * Sigmoid activation
     */
    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + exp(-x.coerceIn(-88f, 88f)))
    }

    /**
     * Normalize angle to [-π, π]
     */
    private fun normalizeRadians(angle: Float): Float {
        val twoPi = 2 * PI.toFloat()
        return angle - twoPi * floor((angle + PI.toFloat()) / twoPi)
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
        Log.d(TAG, "✓ HandDetector closed")
    }
}

/**
 * Hand detection result
 */
data class HandDetection(
    val roi: HandROI,
    val confidence: Float,
    val keypoints: Array<FloatArray>
)

/**
 * Hand ROI (Region of Interest)
 */
data class HandROI(
    val rotation: Float,
    val centerX: Float,
    val centerY: Float,
    val width: Float,
    val height: Float,
    val rectPoints: Array<FloatArray>,
    val frameWidth: Int,
    val frameHeight: Int
)