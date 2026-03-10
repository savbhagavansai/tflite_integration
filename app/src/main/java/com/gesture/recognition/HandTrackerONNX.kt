package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

/**
 * Complete Hand Tracking Pipeline using ONNX models
 *
 * Two-stage pipeline with detection/tracking state machine:
 *
 * DETECTION MODE (first frame or after loss):
 *   1. Run HandDetector on full frame
 *   2. Extract landmarks from detected ROI
 *   3. Switch to tracking mode
 *
 * TRACKING MODE (most frames):
 *   1. Build ROI from previous landmarks (no detector!)
 *   2. Extract landmarks from ROI
 *   3. Continue tracking or revert to detection if lost
 *
 * Performance:
 *   - Detection mode: ~6-9ms (detector + landmarks)
 *   - Tracking mode: ~3-5ms (landmarks only)
 *   - Average: ~4-6ms (mostly tracking)
 */
class HandTrackerONNX(private val context: Context) {

    companion object {
        private const val TAG = "HandTrackerONNX"
        private const val LANDMARK_SMOOTHING_ALPHA = 0.5f  // EMA: 0=frozen, 1=raw, 0.5=balanced
    }

    private val detector: HandDetectorONNX
    private val landmarker: HandLandmarkDetectorONNX

    // State machine
    private var isTracking = false
    private var smoothedLandmarks: Array<FloatArray>? = null
    private var currentROI: HandROI? = null

    // Performance monitoring
    private var lastDetectorTime = 0L
    private var lastLandmarkTime = 0L

    init {
        Log.d(TAG, "Initializing Hand Tracker ONNX...")
        detector = HandDetectorONNX(context)
        landmarker = HandLandmarkDetectorONNX(context)
        Log.d(TAG, "✓ Hand Tracker ONNX ready")
    }

    /**
     * Process frame and extract hand landmarks
     *
     * @param frame Input frame (any size)
     * @return HandTrackingResult or null if no hand
     */
    fun processFrame(frame: Bitmap): HandTrackingResult? {
        val frameWidth = frame.width
        val frameHeight = frame.height

        var roi: HandROI? = null
        var detectorMs = 0f
        var landmarkMs = 0f

        // ── STAGE 1: Get ROI (detection or tracking) ──
        if (!isTracking) {
            // DETECTION MODE: Run full detector
            val t0 = System.nanoTime()
            val detection = detector.detectHand(frame)
            detectorMs = (System.nanoTime() - t0) / 1_000_000f

            if (detection == null) {
                Log.d(TAG, "[DETECT] No hand found")
                return null
            }

            roi = detection.roi
            Log.d(TAG, "[DETECT] Hand found: conf=${detection.confidence}")

        } else {
            // TRACKING MODE: Use previous landmarks for ROI
            val prevLandmarks = smoothedLandmarks
            if (prevLandmarks != null) {
                roi = HandTrackingROI.buildTrackingROI(
                    prevLandmarks,
                    frameWidth,
                    frameHeight
                )
                detectorMs = 0f  // Detector not used!
                Log.d(TAG, "[TRACK] Using previous landmarks for ROI")
            } else {
                // Shouldn't happen, but fallback to detection
                isTracking = false
                return processFrame(frame)
            }
        }

        currentROI = roi

        // ── STAGE 2: Extract landmarks from ROI ──
        if (roi != null) {
            val t0 = System.nanoTime()
            val result = landmarker.detectLandmarks(frame, roi)
            landmarkMs = (System.nanoTime() - t0) / 1_000_000f

            if (result == null) {
                // Hand lost → reset to detection mode
                Log.d(TAG, "[LOST] Hand presence too low → back to detector")
                isTracking = false
                smoothedLandmarks = null
                currentROI = null
                return null
            }

            // Apply EMA smoothing
            val rawLandmarks = result.landmarks
            if (smoothedLandmarks == null) {
                // First frame: initialize
                smoothedLandmarks = rawLandmarks.map { it.copyOf() }.toTypedArray()
            } else {
                // Smooth only x, y (keep z raw)
                for (i in rawLandmarks.indices) {
                    smoothedLandmarks!![i][0] =
                        LANDMARK_SMOOTHING_ALPHA * rawLandmarks[i][0] +
                        (1 - LANDMARK_SMOOTHING_ALPHA) * smoothedLandmarks!![i][0]

                    smoothedLandmarks!![i][1] =
                        LANDMARK_SMOOTHING_ALPHA * rawLandmarks[i][1] +
                        (1 - LANDMARK_SMOOTHING_ALPHA) * smoothedLandmarks!![i][1]

                    smoothedLandmarks!![i][2] = rawLandmarks[i][2]  // z unchanged
                }
            }

            // Enable tracking for next frame
            isTracking = true

            // Store timing
            lastDetectorTime = detectorMs.toLong()
            lastLandmarkTime = landmarkMs.toLong()

            return HandTrackingResult(
                landmarks = smoothedLandmarks!!,
                presence = result.presence,
                handedness = result.handedness,
                roi = roi,
                detectorTimeMs = detectorMs,
                landmarkTimeMs = landmarkMs,
                totalTimeMs = detectorMs + landmarkMs,
                wasTracking = isTracking
            )
        }

        return null
    }

    /**
     * Get last smoothed landmarks (for display or gesture recognition)
     */
    fun getLastLandmarks(): FloatArray? {
        val landmarks = smoothedLandmarks ?: return null

        // Flatten to (21 × 3) = 63 values
        val flat = FloatArray(63)
        for (i in 0 until 21) {
            flat[i * 3] = landmarks[i][0]
            flat[i * 3 + 1] = landmarks[i][1]
            flat[i * 3 + 2] = landmarks[i][2]
        }
        return flat
    }

    /**
     * Check if currently tracking
     */
    fun isCurrentlyTracking(): Boolean = isTracking

    /**
     * Get last detector time (ms)
     */
    fun getLastDetectorTimeMs(): Float = lastDetectorTime.toFloat()

    /**
     * Get last landmark time (ms)
     */
    fun getLastLandmarkTimeMs(): Float = lastLandmarkTime.toFloat()

    /**
     * Reset tracking state (force detection on next frame)
     */
    fun resetTracking() {
        isTracking = false
        smoothedLandmarks = null
        currentROI = null
        Log.d(TAG, "Tracking reset")
    }

    fun close() {
        detector.close()
        landmarker.close()
        Log.d(TAG, "Hand Tracker ONNX closed")
    }
}

/**
 * Hand tracking result
 */
data class HandTrackingResult(
    val landmarks: Array<FloatArray>,  // 21 landmarks × (x, y, z) smoothed
    val presence: Float,
    val handedness: String,
    val roi: HandROI,
    val detectorTimeMs: Float,
    val landmarkTimeMs: Float,
    val totalTimeMs: Float,
    val wasTracking: Boolean
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as HandTrackingResult

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