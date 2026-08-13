def areSimilar(drafts, submissions):
    results = []
    
    for i in range(len(drafts)):
        d = drafts[i]
        s = submissions[i]
    
        dif = {}
        for j in d:
            if j not in dif:
                dif[j] = 1
            else:
                dif[j] += 1
        
        dif2 = {}
        for k in s:
            if k not in dif2:
                dif2[k] = 1
            else:
                dif2[k] += 1
        
        all_keys = set(dif2) | set(dif)
        no = 0
        for key in all_keys:
            v1 = dif.get(key, 0)
            v2 = dif2.get(key, 0)
            if abs(v1 - v2) > 3:
                no = 1 
                break
        
        if no == 1: 
            results.append("NO")
        else:
            results.append("YES")
    
    return results