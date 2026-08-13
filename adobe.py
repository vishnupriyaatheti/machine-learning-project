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


def getUserNameStrength(usernames, common_words):
    restricted_set = set(common_words)
    results = []
    
    for username in usernames:
        is_weak = False
        
        if len(username) < 6:
            is_weak = True
            
        if not is_weak and username.isdigit():
            is_weak = True
        
        if not is_weak and (username.isupper() or username.islower()):
            is_weak = True
        
        if not is_weak and username in restricted_set:
            is_weak = True
        
        elif not is_weak:
            for word in common_words:
                if word in username:
                    is_weak = True
                    break

        results.append("weak" if is_weak else "strong")
    
    return results


def shopping_total(items):

    
    apply_discount = lambda price: price * 0.9 if price > 50 else price
    
    
    total = sum(map(apply_discount, (price for name, price in items)))
    
    return total