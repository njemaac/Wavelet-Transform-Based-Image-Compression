package projectDWT.core;
import util.Logger;

public final class MatrixUtil {
    private MatrixUtil(){}

    public static double[][] copy(double[][] m){
        int h = m.length;
        int w = m[0].length;
        double[][] out = new double[h][w];
        for (int y = 0; y < h; y++){
            System.arraycopy(m[y], 0, out[y], 0, w);
        }
        return out;
    }
    public static double maxAbs(double[][] m){
        double max = 0.0;
        for (double[] row:m){
            for (double v:row){
                double a = Math.abs(v);
                if (a>max) max = a;
            }
        }
        return max;
    }
    public static void thresholdInPlace(double[][] m, double thr){
        int h = m.length;
        int w = m[0].length;
        int zeros = 0;
        int total = h*w;

        for (int y = 0; y<h; y++){
            for (int x = 0; x<w; x++){
                if (Math.abs(m[y][x]) < thr){
                    m[y][x] = 0.0;
                    zeros++;
                }
            }
        }
        Logger.info("Thresholding done. Zeroed " + zeros + " / " + total + " coefficients");

    }
    public static int clampToByte(double v){
        int x = (int)Math.round(v);
        if (x<0) return 0;
        if (x>255) return 255;
        return x;
    }
}
