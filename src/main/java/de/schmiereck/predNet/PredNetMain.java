package de.schmiereck.predNet;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class PredNetMain {
    public static void main(String[] args) throws InterruptedException {
        System.out.println("PredNet V1.0.0");

        final PredNetService predNetService = new PredNetService();

        predNetService.calc(); // einmal initial berechnen

        // Hintergrund-Thread: ruft alle 100 ms calc() auf
        final ScheduledExecutorService calcScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "PredNetCalc");
            t.setDaemon(true); // Daemon, damit JVM bei Ende des Hauptthreads beenden kann
            return t;
        });
        calcScheduler.scheduleAtFixedRate(() -> predNetService.calc(), 0, 100, TimeUnit.MILLISECONDS);

        // Zweiter Thread: zeigt alle 40 ms die Kurve an
        final ScheduledExecutorService showScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "PredNetShowCurve");
            t.setDaemon(true); // ebenfalls Daemon
            return t;
        });
        showScheduler.scheduleAtFixedRate(() -> showCurve(predNetService), 0, 40, TimeUnit.MILLISECONDS);

        // Hauptthread bleibt eine Zeit lang aktiv, damit Ausgaben sichtbar sind
        Thread.sleep(2000); // 2 Sekunden Demo-Lauf

        // Scheduler beenden
        calcScheduler.shutdownNow();
        showScheduler.shutdownNow();
    }

    private static void showCurve(PredNetService predNetService) {
        final CurveDto curveDto = predNetService.retrieveCurve();
        final int[] inputArr = curveDto.getInputArr();
        for (int xPos = 0; xPos < inputArr.length; xPos++) {
            System.out.printf("%3d ", inputArr[xPos]);
        }
        System.out.printf(" : %3d", curveDto.getOutput());
        System.out.println();
    }
}
