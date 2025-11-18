package de.schmiereck.predNet;

public class CurveDto {
    private final int[] inputArr;
    private final int output;

    public CurveDto(final int[] inputArr, final int output) {
        this.inputArr = inputArr;
        this.output = output;
    }

    public int[] getInputArr() {
        return this.inputArr;
    }

    public int getOutput() {
        return this.output;
    }
}
