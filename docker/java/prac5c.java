class prac5c {
    public static void main(String args[]) {
        int n1 = 0, n2 = 1, n3, i, count = 15;
        System.out.print(n1 + " " + n2);

        for (i = 2; i < count; ++i) {
            n3 = n1 + n2;
            System.out.print(" " + n3);
            n1 = n2;
            n2 = n3;
        }
        System.out.println("");
        System.out.println("Abdulkader Kanchwala 033");
    }
}

//docker build -f Dockerfile -t prac5c .
//docker run prac5c
