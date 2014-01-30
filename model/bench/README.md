##pair analysis of errors

Here is the case by case analysis of wrongly classified domain pairs for the naive bayes model. The notation(d1, d2, c) means there are c cases in the testing data where our model predicted d1 while the actual domain is d2.

### (Web, Note, 73)

    念念 将 太 无 二 的 便签
Word segmentation wrong for '将太无二'. Fixed when seg is right.

    找出 我 与 外婆 家 有关 的 记事
    web -58.1265109079 1
    找出(找出, 726): -10.2843    我(我, 245831): -3.9606	与(与, 4604): -7.5232	外婆(外婆, 84): -10.0432	家(__LOCATION__, 7924): -2.9422	有关(有关, 2004): -9.8490	的(的, 236835): -3.6335	记事(记事, 6038): -10.8905
    ----------------------------------------------------------------
    note -61.5606162708 1
    找出(找出, 726): -8.2810	我(我, 245831): -2.7124	与(与, 4604): -6.9623	外婆(外婆, 84): -28.2644	家(__LOCATION__, 7924): -5.1126	有关(有关, 2004): -5.3422	的(的, 236835): -2.1340	记事(记事, 6038): -3.7517
    
‘记事’ acutally works pretty well. But the effect got undermined by other words like '外婆', which got much lower probability yet larger difference among them. 

I tried to use sqrt to smooth the probabilities. CV accuracy is unaffected but testing accuracy got went up to 87.2%. However there are other cases like 

    今天 又 热 又 潮 吗

Where ‘今天’ and ‘热' favors weather domain, yet ’又’ and other words flooded too much noise.
    
    