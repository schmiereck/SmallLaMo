package de.schmiereck.predNet;

import java.util.Arrays;

public class PredNetService {
    private final int[] precalcCurveArr = new int[]
            {
                    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                    90, 80, 70, 60, 50, 40, 30, 20, 10,
            };
    private volatile int xPosCurve; // volatile für Sichtbarkeit zwischen Threads
    private final int curveLength;
    private volatile int[] inputCurveArr; // volatile Referenz, wird in calc() neu erzeugt
    private volatile int output; // volatile Referenz, wird in calc() neu erzeugt

    public PredNetService() {
        this.xPosCurve = 0;
        this.curveLength = 10;
        this.inputCurveArr = new int[this.curveLength];
        this.output = 0;
    }

    public CurveDto retrieveCurve() {
        // lokale Kopie der aktuellen Referenz
        final int[] currentArr = this.inputCurveArr;
        // neue Kopie für den DTO (Isolation vom Hintergrund-Array)
        final int[] inputArr = Arrays.copyOf(currentArr, this.curveLength);
        return new CurveDto(inputArr, this.output);
    }

    public void calc() {
        // Neues Array aufbauen (Copy-on-Write) statt In-Place Mutation
        final int[] newArr = new int[this.curveLength];
        for (int posX = 0; posX < this.curveLength; posX++) {
            newArr[posX] = this.precalcCurveArr[(this.xPosCurve + posX) % this.precalcCurveArr.length];
        }
        // Nächste Position
        final int nextXPos = (this.xPosCurve + 1) % this.precalcCurveArr.length;
        // Veröffentlichung: zuerst das neue Array, dann den Index (oder umgekehrt; hier egal, beide volatile)
        this.inputCurveArr = newArr;
        this.xPosCurve = nextXPos;

        this.output = newArr[newArr.length - 1];
    }
}
