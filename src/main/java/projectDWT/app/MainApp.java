package projectDWT.app;

import projectDWT.distributed.DistributedWaveletMPJ;
import util.Logger;
import mpi.*;

import javax.swing.*;

public class MainApp {

    public static void main(String[] args) throws Exception {

        String mode = resolveMode(args);

        Logger.info("=== DWT Image Compression ===");
        Logger.info("Mode     : " + mode.toUpperCase());
        Logger.info("CPU cores: " + Runtime.getRuntime().availableProcessors());
        Logger.info("Max RAM  : " + (Runtime.getRuntime().maxMemory() / 1024 / 1024) + " MB");

        switch (mode) {
            case "sequential" -> SwingUtilities.invokeLater(WaveletSequentialApp::new);
            case "parallel"   -> SwingUtilities.invokeLater(WaveletParallelApp::new);

            case "distributed" -> {
                MPI.Init(args);
                int rank = MPI.COMM_WORLD.Rank();
                boolean isRoot = (rank == 0);

                DistributedWaveletMPJ.runMPI(args, isRoot);

                MPI.Finalize();
            }

            default -> {
                Logger.error("Unknown mode: " + mode);
                System.exit(1);
            }
        }
    }

    private static String resolveMode(String[] args) {
        for (String a : args) {
            String lower = a.toLowerCase();
            if (lower.equals("sequential")) return "sequential";
            if (lower.equals("parallel"))   return "parallel";
            if (lower.equals("distributed")) return "distributed";
        }
        String[] options = {"Sequential", "Parallel"};
        int choice = JOptionPane.showOptionDialog(
                null,
                "DWT Image Compression\n\nSelect execution mode:",
                "Mode Selection",
                JOptionPane.DEFAULT_OPTION,
                JOptionPane.QUESTION_MESSAGE,
                null,
                options,
                options[0]);

        return switch (choice) {
            case 0 -> "sequential";
            case 1 -> "parallel";
            default -> "sequential";
        };
    }
}