package projectDWT.core;

import java.awt.image.BufferedImage;

public interface WaveletPipeline {
    BufferedImage compressAndReconstruct(BufferedImage img, int thresholdPercent) throws Exception;
}