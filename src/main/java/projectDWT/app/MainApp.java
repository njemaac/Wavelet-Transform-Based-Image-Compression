package projectDWT.app;

import mpi.MPI;
import projectDWT.distributed.DistributedWaveletMPJ;
import util.Logger;

public class MainApp {

    public static void main(String[] args) throws Exception {

        boolean isDistributed = false;
        for (String arg : args) {
            if (arg.toLowerCase().contains("distributed")) {
                isDistributed = true;
                break;
            }
        }

        Logger.info("=== DWT Image Compression ===");
        Logger.info("Mode detected: " + (isDistributed ? "DISTRIBUTED" : "GUI (Sequential/Parallel)"));

        if (isDistributed) {
            MPI.Init(args);
            int rank = MPI.COMM_WORLD.Rank();
            boolean isRoot = (rank == 0);

            Logger.info("[Rank " + rank + "] MPI initialized successfully");

            if (isRoot) {
                Logger.info("Root process starting distributed compression...");
            }

            DistributedWaveletMPJ.runMPI(args, isRoot);

            MPI.Finalize();
            return;
        }
        new WaveletApp();
    }
}