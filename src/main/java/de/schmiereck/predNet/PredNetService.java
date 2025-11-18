package de.schmiereck.predNet;

public class PredNetService {
    private int[] precalcCurveArr = new int[]
            {
                    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                    90, 80, 70, 60, 50, 40, 30, 20, 10,
            };
    private int xPosCurve = 0;
    private int curveLength = 10;
    private int[] inputCurveArr = new int[curveLength];

    public CurveDto retrieveCurve() {
        final int[] inputArr = new int[this.curveLength];
        for (int posX = 0; posX < this.curveLength; posX++) {
            inputArr[posX] = this.inputCurveArr[posX];
        }
        final CurveDto curveDto = new CurveDto(inputArr);
        return curveDto;
    }

    public void calc() {
        for (int posX = 0; posX < this.curveLength; posX++) {
            this.inputCurveArr[posX] = this.precalcCurveArr[(this.xPosCurve + posX) % this.precalcCurveArr.length];
        }
        final int nextXPos = (this.xPosCurve + 1) % this.precalcCurveArr.length;
        this.xPosCurve = nextXPos;
    }
}
