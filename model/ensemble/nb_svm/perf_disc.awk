BEGIN {
    corrected = 0;
    wrong = 0;
    protential = 0;
    both_right = 0;
    both_wrong = 0;
    print "" > "disc_corrected.dat";
    print "" > "disc_wrong.dat";
}

{
   front = $2;
   alt = $3;
   back = $4;
   gold = $5;
   if (front != back && back == gold) {
        ++corrected;
        print $0 >> "disc_corrected.dat";
   }
   if (front != back && front == gold) {
        ++wrong;
        print $0 >> "disc_wrong.dat";
   }
   if (alt == gold) {
       ++protential;
   }
   if (front != gold && alt != gold) {
       ++both_wrong;
   }
   if (front == gold && alt == gold) {
       ++both_right;
   }
}

END {
    printf("protential=%s, correct=%d, wrong=%d, both_right=%d, both_wrong=%d\n",
           protential, corrected, wrong, both_right, both_wrong);
}
