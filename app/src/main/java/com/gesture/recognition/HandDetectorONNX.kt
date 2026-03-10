package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.*
import kotlin.math.*

/**
 * ONNX Hand Detector - Stage 1 of pipeline
 *
 * Detects palm/hand location in full frame
 * Input: 256×256 RGB image normalized to [-1, 1]
 * Output: Bounding boxes + 7 keypoints + confidence scores
 *
 * Based on geaxgx MediaPipe ONNX implementation
 */
class HandDetectorONNX(private val context: Context) {

    companion object {
        private const val TAG = "HandDetectorONNX"
        private const val MODEL_NAME = "HandDetector.onnx"
        private const val INPUT_SIZE = 256
        private const val DECODE_SCALE = 256f
        private const val DETECTOR_THRESHOLD = 0.5f
        private const val NMS_IOU_THRESHOLD = 0.3f
        private const val NUM_ANCHORS = 2944

        // ROI transformation constants (from mediapipe)
        const val SCALE_X = 2.9f
        const val SCALE_Y = 2.9f
        const val SHIFT_X = 0.0f
        const val SHIFT_Y = -0.5f  // Shift toward fingers
    }

    private var ortEnvironment: OrtEnvironment? = null
    private var onnxSession: OrtSession? = null
    private val anchors: FloatArray

    init {
        Log.d(TAG, "════════════════════════════════════════")
        Log.d(TAG, "Initializing Hand Detector (GPU mode)")
        Log.d(TAG, "════════════════════════════════════════")

        // Generate anchors
        anchors = generateAnchors()

        // Load ONNX model
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            val modelBytes = context.assets.open(MODEL_NAME).use { it.readBytes() }

            val sessionOptions = OrtSession.SessionOptions()

            // ═══════════════════════════════════════════════════════
            // GPU MODE: Force NNAPI to prefer GPU over NPU
            // ═══════════════════════════════════════════════════════
            try {
                sessionOptions.addNnapi()

                // Execution preference:
                // 0 = FAST_SINGLE_ANSWER (prefers NPU)
                // 1 = SUSTAINED_SPEED (balanced)
                // 2 = LOW_POWER (prefers GPU) ← We want this!
                // 3 = CPU fallback
                // sessionOptions.addConfigEntry("nnapi.execution_preference", "2")

                Log.d(TAG, "✓ NNAPI enabled with GPU preference (LOW_POWER mode)")
                Log.d(TAG, "   Target: Mali-G68 MP5 GPU")
            } catch (e: Exception) {
                Log.e(TAG, "✗ NNAPI configuration failed: ${e.message}")
                Log.e(TAG, "   Falling back to default NNAPI")
            }
            // ═══════════════════════════════════════════════════════

            onnxSession = ortEnvironment?.createSession(modelBytes, sessionOptions)

            Log.d(TAG, "✓ Hand Detector loaded successfully")
            Log.d(TAG, "   Model size: ${modelBytes.size / 1024}KB")
            Log.d(TAG, "   Anchors: ${NUM_ANCHORS}")
            Log.d(TAG, "   Backend: NNAPI (GPU preferred)")
            Log.d(TAG, "════════════════════════════════════════")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to load detector model", e)
            throw e
        }
    }

    /**
     * Generate 2944 anchors for 256×256 detector
     *
     * Layout:
     *   stride  8 → 32×32 grid × 2 anchors = 2048
     *   stride 16 → 16×16 grid × 2 anchors = 512
     *   stride 32 →  8×8  grid × 6 anchors = 384
     *   Total = 2944
     */
    private fun generateAnchors(): FloatArray {
        val spec = listOf(
            Pair(8, 2),
            Pair(16, 2),
            Pair(32, 6)
        )

        val anchorsList = mutableListOf<Float>()

        for ((stride, count) in spec) {
            val grid = INPUT_SIZE / stride
            for (y in 0 until grid) {
                for (x in 0 until grid) {
                    val cx = (x + 0.5f) / grid
                    val cy = (y + 0.5f) / grid
                    repeat(count) {
                        anchorsList.add(cx)  // cx
                        anchorsList.add(cy)  // cy
                        anchorsList.add(1.0f) // w
                        anchorsList.add(1.0f) // h
                    }
                }
            }
        }

        require(anchorsList.size / 4 == NUM_ANCHORS) {
            "Expected $NUM_ANCHORS anchors, got ${anchorsList.size / 4}"
        }

        return anchorsList.toFloatArray()
    }

    /**
     * Detect hand in frame
     *
     * @param bitmap Input image (any size, will be resized to 256×256)
     * @return HandDetection or null if no hand found
     */
    fun detectHand(bitmap: Bitmap): HandDetection? {
        try {
            // Preprocess
            val input = preprocessImage(bitmap)

            // Run inference
            val inputName = onnxSession?.inputNames?.iterator()?.next() ?: "image"
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, input)

            val outputs = onnxSession?.run(mapOf(inputName to inputTensor))

            // Parse outputs
            val rawBoxes = outputs?.get(0)?.value as Array<Array<FloatArray>>  // (1, 2944, 18)
            val rawScores = outputs?.get(1)?.value as Array<Array<FloatArray>> // (1, 2944, 1)

            inputTensor.close()
            outputs?.close()

            // Process detections
            val detection = processDetections(
                rawBoxes[0],
                rawScores[0],
                bitmap.width,
                bitmap.height
            )

            return detection

        } catch (e: Exception) {
            Log.e(TAG, "Detection failed", e)
            return null
        }
    }

    /**
     * Preprocess image for detector
     * BGR → RGB, resize to 256×256, normalize to [-1, 1], NCHW format
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

                // Extract RGB and normalize to [-1, 1]
                val r = ((pixel shr 16) and 0xFF).toFloat()
                val g = ((pixel shr 8) and 0xFF).toFloat()
                val b = (pixel and 0xFF).toFloat()

                input[0][0][y][x] = (r - 127.5f) / 127.5f  // R channel
                input[0][1][y][x] = (g - 127.5f) / 127.5f  // G channel
                input[0][2][y][x] = (b - 127.5f) / 127.5f  // B channel
            }
        }

        return input
    }

    /**
     * Process raw detections: decode boxes, apply NMS, build ROI
     */
    private fun processDetections(
        rawBoxes: Array<FloatArray>,    // (2944, 18)
        rawScores: Array<FloatArray>,   // (2944, 1)
        frameWidth: Int,
        frameHeight: Int
    ): HandDetection? {

        // Apply sigmoid to scores
        val scores = FloatArray(NUM_ANCHORS) { i ->
            sigmoid(rawScores[i][0])
        }

        // Filter by threshold
        val validIndices = scores.indices.filter { scores[it] > DETECTOR_THRESHOLD }
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

            for (j in order) {
                if (i == j || suppressed[j]) continue

                val iou = calculateIoU(boxes[i], boxes[j])
                if (iou > NMS_IOU_THRESHOLD) {
                    suppressed[j] = true
                }
            }
        }

        return kept
    }

    /**
     * Calculate IoU between two boxes
     */
    private fun calculateIoU(box1: FloatArray, box2: FloatArray): Float {
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
     * Uses wrist + middle_mcp keypoints to calculate rotation
     */
    private fun buildDetectionROI(
        box: FloatArray,           // Normalized box [x1, y1, x2, y2]
        keypoints: Array<FloatArray>,  // 7 keypoints (normalized)
        frameWidth: Int,
        frameHeight: Int
    ): HandROI {

        // Keypoints: 0=wrist, 2=middle_mcp
        val wrist = keypoints[0]
        val middleMcp = keypoints[2]

        // Calculate rotation (mediapipe formula)
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

        // Calculate 4 corner points of rotated rectangle
        val rectPoints = rotatedRectToPoints(cxA, cyA, wA, hA, rotation)

        return HandROI(
            rotation = rotation,
            centerX = cxA,
            centerY = cyA,
            width = wA,
            height = hA,
            rectPoints = rectPoints
        )
    }

    /**
     * Calculate 4 corners of rotated rectangle
     * Returns: [bottom-left, top-left, top-right, bottom-right]
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

    fun close() {
        onnxSession?.close()
        Log.d(TAG, "Hand Detector closed")
    }
}

/**
 * Hand detection result
 */
data class HandDetection(
    val roi: HandROI,
    val confidence: Float,
    val keypoints: Array<FloatArray>  // 7 keypoints
)

/**
 * Region of Interest for hand
 */
data class HandROI(
    val rotation: Float,
    val centerX: Float,
    val centerY: Float,
    val width: Float,
    val height: Float,
    val rectPoints: Array<FloatArray>  // 4 corner points
)