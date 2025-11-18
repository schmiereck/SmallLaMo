package de.schmiereck.predNet;

public class PredNetMain {
    public static void main(String[] args) {
        System.out.println("PredNet V1.0.0");

        final PredNetService predNetService = new PredNetService();

        for (int calcPos = 0; calcPos < 25; calcPos++) {
            predNetService.calc();

            final CurveDto curveDto = predNetService.retrieveCurve();
            final int[] inputArr = curveDto.getInputArr();
            for (int xPos = 0; xPos < inputArr.length; xPos++) {
                System.out.printf("%3d ", inputArr[xPos]);
            }
            System.out.println();
        }
    }
}
