package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.gesture.recognition.GestureResult

/**
 * Complete Gesture Recognition System - HYBRID ARCHITECTURE
 *
 * Pipeline:
 * 1. HandTrackerTFLite: Detect hand + extract 21 landmarks (GPU/NPU via TFLite)
 * 2. LandmarkNormalizer: Normalize landmarks (existing, no changes!)
 * 3. SequenceBuffer: Buffer 15 frames (existing, no changes!)
 * 4. ONNXInference: Classify gesture using TCN (NPU via ONNX)
 *
 * HYBRID: TFLite (hand tracking) + ONNX (gesture model)
 *
 * Expected performance:
 *   - Hand tracking: 12-25ms (TFLite on GPU/NPU)
 *   - Gesture classification: 3-5ms (ONNX on NPU)
 *   - Total: 15-30ms → 33-66 FPS!
 */
class GestureRecognizerHybrid(private val context: Context) {

    companion object {
        private const val TAG = "GestureRecognizerHybrid"
    }

    // Components
    val handTracker: HandTrackerTFLite
    val sequenceBuffer: SequenceBuffer
    val gestureClassifier: ONNXInference

    // State
    private var lastLandmarks: FloatArray? = null

    // Performance tracking
    private var lastGestureTimeMs = 0.0

    init {
        Log.d(TAG, "════════════════════════════════════════")
        Log.d(TAG, "Initializing Gesture Recognizer (Hybrid)")
        Log.d(TAG, "  TFLite: Hand tracking (GPU/NPU)")
        Log.d(TAG, "  ONNX: Gesture model (NPU)")
        Log.d(TAG, "════════════════════════════════════════")

        handTracker = HandTrackerTFLite(context)
        sequenceBuffer = SequenceBuffer(Config.SEQUENCE_LENGTH)
        gestureClassifier = ONNXInference(context)

        Log.d(TAG, "✓ Gesture Recognizer Hybrid ready")
        Log.d(TAG, "  Hand Detector: ${handTracker.getDetectorBackend()}")
        Log.d(TAG, "  Hand Landmarks: ${handTracker.getLandmarkBackend()}")
        Log.d(TAG, "════════════════════════════════════════")
    }

    /**
     * Process frame through complete pipeline
     *
     * @param bitmap Input frame
     * @return GestureResultHybrid or null if no hand detected
     */
    fun processFrame(bitmap: Bitmap): GestureResultHybrid? {
        // ═══════════════════════════════════════════════════════════
        // Stage 1: Hand tracking (TFLite on GPU/NPU)
        // ═══════════════════════════════════════════════════════════
        val trackingResult = handTracker.processFrame(bitmap)

        if (trackingResult == null) {
            // No hand detected
            lastLandmarks = null
            sequenceBuffer.clear()
            lastGestureTimeMs = 0.0
            return null
        }

        // ═══════════════════════════════════════════════════════════
        // Stage 2: Normalize landmarks (existing LandmarkNormalizer)
        // ═══════════════════════════════════════════════════════════
        val landmarksFlat = flattenLandmarks(trackingResult.landmarks)
        val normalizedLandmarks = LandmarkNormalizer.normalize(landmarksFlat)

        // Store for external access
        lastLandmarks = landmarksFlat

        // ═══════════════════════════════════════════════════════════
        // Stage 3: Add to sequence buffer (existing SequenceBuffer)
        // ═══════════════════════════════════════════════════════════
        sequenceBuffer.add(normalizedLandmarks)

        if (!sequenceBuffer.isFull()) {
            // Buffer not full yet, return temp result with timing
            return GestureResultHybrid(
                gestureResult = GestureResult(
                    gesture = "buffering",
                    confidence = 0f,
                    allProbabilities = FloatArray(Config.NUM_CLASSES),
                    handDetectorTimeMs = trackingResult.detectorTimeMs.toDouble(),
                    landmarksTimeMs = trackingResult.landmarkTimeMs.toDouble(),
                    gestureTimeMs = 0.0,
                    totalTimeMs = trackingResult.totalTimeMs.toDouble(),
                    wasTracking = trackingResult.wasTracking
                ),
                landmarks = landmarksFlat,
                handTracking = trackingResult,
                bufferSize = sequenceBuffer.size()
            )
        }

        // ═══════════════════════════════════════════════════════════
        // Stage 4: Gesture classification (ONNX on NPU)
        // ═══════════════════════════════════════════════════════════
        val sequence = sequenceBuffer.getSequence()

        if (sequence == null) {
            return GestureResultHybrid(
                gestureResult = null,
                landmarks = landmarksFlat,
                handTracking = trackingResult,
                bufferSize = sequenceBuffer.size()
            )
        }

        // Run TCN gesture classifier
        val t0 = System.nanoTime()
        val gestureOutput = gestureClassifier.classify(sequence)
        val gestureMs = (System.nanoTime() - t0) / 1_000_000.0

        lastGestureTimeMs = gestureMs

        // Create gesture result
        val gestureResult = GestureResult(
            gesture = gestureOutput.gestureName,
            confidence = gestureOutput.confidence,
            allProbabilities = gestureOutput.allProbabilities,
            handDetectorTimeMs = trackingResult.detectorTimeMs.toDouble(),
            landmarksTimeMs = trackingResult.landmarkTimeMs.toDouble(),
            gestureTimeMs = gestureMs,
            totalTimeMs = trackingResult.totalTimeMs + gestureMs,
            wasTracking = trackingResult.wasTracking
        )

        return GestureResultHybrid(
            gestureResult = gestureResult,
            landmarks = landmarksFlat,
            handTracking = trackingResult,
            bufferSize = sequenceBuffer.size()
        )
    }

    /**
     * Flatten landmarks from [21, 3] to [63]
     */
    private fun flattenLandmarks(landmarks: Array<FloatArray>): FloatArray {
        val flattened = FloatArray(63)
        for (i in landmarks.indices) {
            flattened[i * 3] = landmarks[i][0]
            flattened[i * 3 + 1] = landmarks[i][1]
            flattened[i * 3 + 2] = landmarks[i][2]
        }
        return flattened
    }

    /**
     * Get last processed landmarks
     */
    fun getLastLandmarks(): FloatArray? = lastLandmarks

    /**
     * Get buffer fill percentage
     */
    fun getBufferFillPercentage(): Float {
        return sequenceBuffer.size().toFloat() / Config.SEQUENCE_LENGTH
    }

    /**
     * Reset all state
     */
    fun reset() {
        handTracker.reset()
        sequenceBuffer.clear()
        lastLandmarks = null
        lastGestureTimeMs = 0.0
        Log.d(TAG, "Gesture recognizer reset")
    }

    /**
     * Get backend information
     */
    fun getHandTrackingBackend(): String {
        return "Detector: ${handTracker.getDetectorBackend()}, " +
               "Landmarks: ${handTracker.getLandmarkBackend()}"
    }

    /**
     * Get gesture backend information
     */
    fun getGestureBackend(): String {
        return gestureClassifier.getAccelerator()
    }

    /**
     * Release resources
     */
    fun close() {
        handTracker.close()
        gestureClassifier.close()
        Log.d(TAG, "✓ Gesture Recognizer closed")
    }
}

/**
 * Complete gesture recognition result (hybrid)
 */
data class GestureResultHybrid(
    val gestureResult: GestureResult?,
    val landmarks: FloatArray?,
    val handTracking: HandTrackingResult,
    val bufferSize: Int
)
