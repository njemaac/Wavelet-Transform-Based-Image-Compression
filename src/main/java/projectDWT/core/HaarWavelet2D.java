package projectDWT.core;

import java.util.ArrayList;
import java.util.List;

public class HaarWavelet2D {

    public float[][] forward(float[][] src, int maxLevels) {
        float[][] a = MatrixUtil.copy(src);
        horizontalForwardInPlace(a, maxLevels);
        verticalForwardInPlace(a, maxLevels);
        return a;
    }

    public float[][] inverse(float[][] coef, int maxLevels) {
        float[][] a = MatrixUtil.copy(coef);
        verticalInverseInPlace(a, maxLevels);   //vertical first
        horizontalInverseInPlace(a, maxLevels);
        return a;
    }

    //horizontal DWT
    public void horizontalForwardInPlace(float[][] mat, int maxLevels) {
        int h = mat.length;
        int w = mat[0].length;
        int curW = w;
        int level = 0;
        while (curW >= 2 && (maxLevels == -1 || level < maxLevels)) {
            float[] row = new float[curW];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < curW; x++) row[x] = mat[y][x];
                haar1DInPlace(row, curW);
                for (int x = 0; x < curW; x++) mat[y][x] = row[x];
            }
            curW /= 2;
            level++;
        }
    }

    public void horizontalInverseInPlace(float[][] mat, int maxLevels) {
        int h = mat.length;
        int w = mat[0].length;
        List<Integer> widths = new ArrayList<>();
        int curW = w;
        int level = 0;
        while (curW >= 2 && (maxLevels == -1 || level < maxLevels)) {
            widths.add(curW);
            curW /= 2;
            level++;
        }
        for (int i = widths.size() - 1; i >= 0; i--) {
            int regW = widths.get(i);
            float[] row = new float[regW];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < regW; x++) row[x] = mat[y][x];
                inverseHaar1DInPlace(row, regW);
                for (int x = 0; x < regW; x++) mat[y][x] = row[x];
            }
        }
    }

    //vertical DWT
    public void verticalForwardInPlace(float[][] mat, int maxLevels) {
        int h = mat.length;
        int w = mat[0].length;
        int curH = h;
        int level = 0;
        while (curH >= 2 && (maxLevels == -1 || level < maxLevels)) {
            float[] col = new float[curH];
            for (int x = 0; x < w; x++) {
                for (int y = 0; y < curH; y++) col[y] = mat[y][x];
                haar1DInPlace(col, curH);
                for (int y = 0; y < curH; y++) mat[y][x] = col[y];
            }
            curH /= 2;
            level++;
        }
    }

    public void verticalInverseInPlace(float[][] mat, int maxLevels) {
        int h = mat.length;
        int w = mat[0].length;
        List<Integer> heights = new ArrayList<>();
        int curH = h;
        int level = 0;
        while (curH >= 2 && (maxLevels == -1 || level < maxLevels)) {
            heights.add(curH);
            curH /= 2;
            level++;
        }
        for (int i = heights.size() - 1; i >= 0; i--) {
            int regH = heights.get(i);
            float[] col = new float[regH];
            for (int x = 0; x < w; x++) {
                for (int y = 0; y < regH; y++) col[y] = mat[y][x];
                inverseHaar1DInPlace(col, regH);
                for (int y = 0; y < regH; y++) mat[y][x] = col[y];
            }
        }
    }

    //private methods
    private void haar1DInPlace(float[] v, int n) {
        if(n < 2) return;

        int lowSize = (n+1)/2;
        float[] tmp = new float[n];
        int i = 0;

        for (int j = 0; j < n - 1; j+=2) {
            float a = v[j];
            float b = v[j+1];
            tmp[i] = (float) ((a + b) / 2.0);
            tmp[i + lowSize] = (float) ((a - b) / 2.0);
            i++;
        }

        if( n%2 == 1){
            tmp[i] = v[n - 1];
        }
        System.arraycopy(tmp, 0, v, 0, n);
    }

    private void inverseHaar1DInPlace(float[] v, int n) {
        if (n < 2) return;

        int lowSize = (n+1) / 2;
        float[] tmp = new float[n];

        //reconstructing all of the full pairs
        for (int i = 0; i < n/2; i++) {
            float avg = v[i];
            float diff = v[i + lowSize];
            tmp[2 * i] = avg + diff;
            tmp[2 * i + 1] = avg - diff;
        }

        if (n % 2 == 1){
            tmp[n - 1] = v[lowSize - 1];
        }
        System.arraycopy(tmp, 0, v, 0, n);
    }
}