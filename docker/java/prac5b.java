public class prac5b {
    // Abdulkader Kanchwala TYBSCIT 033
    public static void main(String[] args) {
        int number = 11; 
        long factorial = calculateFactorial(number);
        System.out.println("Factorial of " + number + " is: " + factorial);
        System.out.println("Abdulkader Kanchwala 033");
    }
    public static long calculateFactorial(int n) {
        long result = 1;
        for (int i = 1; i <= n; i++) {
            result *= i;
        }
        return result;
    }

}

//docker build -f Dockerfile -t prac5b .
//docker run prac5a
