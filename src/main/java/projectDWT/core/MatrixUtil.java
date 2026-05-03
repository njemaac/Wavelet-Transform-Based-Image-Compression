package projectDWT.core;
import util.Logger;

public final class MatrixUtil {
    private MatrixUtil(){}

    public static float[][] copy(float[][] m){
        int h = m.length;
        int w = m[0].length;
        float[][] out = new float[h][w];
        for (int y = 0; y < h; y++){
            System.arraycopy(m[y], 0, out[y], 0, w);
        }
        return out;
    }
    public static float maxAbs(float[][] m){
        double max = 0.0;
        for (float[] row:m){
            for (float v:row){
                float a = Math.abs(v);
                if (a>max) max = a;
            }
        }
        return (float) max;
    }
    public static void thresholdInPlace(float[][] m, float thr){
        int h = m.length;
        int w = m[0].length;
        int zeros = 0;
        int total = h*w;

        for (int y = 0; y<h; y++){
            for (int x = 0; x<w; x++){
                if (Math.abs(m[y][x]) < thr){
                    m[y][x] = 0.0F;
                    zeros++;
                }
            }
        }
        Logger.info("Thresholding done. Zeroed " + zeros + " / " + total + " coefficients");

    }
    public static int clampToByte(float v){
        int x = Math.round(v);
        if (x<0) return 0;
        if (x>255) return 255;
        return x;
    }
}
