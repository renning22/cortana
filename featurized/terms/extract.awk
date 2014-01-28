{
    n=split($1, terms, " ");
    for (i=1; i<=n; ++i) {
        printf("%s\n",terms[i]);
    }
}
