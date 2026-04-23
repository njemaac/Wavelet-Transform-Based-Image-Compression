package projectDWT.core;

public class TimerMs {
    private long start;

    public void start(){
        start = System.currentTimeMillis();
    }

    public long stop(){
        return System.currentTimeMillis() - start;
    }
}
