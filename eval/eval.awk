BEGIN {
    printf("%8s\t%8s\n", "Predict", "Gold") > "error.dat";
    printf("%8s\t%8s\n", "Predict", "Gold") > "error_pair.dat.tmp";
}

{
    ++predicted[$2];
    ++gold[$3];
    ++total;
    if ($2 == $3) {
        ++correct[$2];
        ++total_correct;
    } else {
        # record the error
        printf("%8s\t%8s\t%s\n", $2, $3, $1) >> "error.dat";
        ++miss[$2,$3];
    }
}

END {
    printf("%8s\t%8s\t%8s\t%8s\n", "Domain", "Predicted", "Recall", "F1-Score");
    for (domain in gold) {
        precision = correct[domain] / predicted[domain]
        recall = correct[domain] / gold[domain]
        f1 = 2 * precision * recall / (precision + recall)
        printf("%8s\t%f\t%f\t%f\n", domain, precision, recall, f1);
    }
    for (pair in miss) {
        split(pair, p, SUBSEP);
        printf("%8s\t%8s\t%d\n", p[1], p[2], miss[pair]) >> "error_pair.dat.tmp";
    }
    printf("Total accuracy\t%f\n", total_correct / total);
}