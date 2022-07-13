import numpy as np


def levenshtein_distance(hypothesis: list, reference: list):
    """编辑距离
    计算两个序列的levenshtein distance，可用于计算 WER/CER
    参考资料：
        https://www.cuelogic.com/blog/the-levenshtein-algorithm
        https://martin-thoma.com/word-error-rate-calculation/

    C: correct
    W: wrong
    I: insert
    D: delete
    S: substitution

    :param hypothesis: 预测序列
    :param reference: 真实序列
    :return: 1: 错误操作，所需要的 S，D，I 操作的次数;
             2: ref 与 hyp 的所有对齐下标
             3: 返回 C、W、S、D、I 各自的数量
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    # 记录所有的操作，0-equal；1-insertion；2-deletion；3-substitution
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    # 生成 cost 矩阵和 operation矩阵，i:外层hyp，j:内层ref
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1

                # compare_val = [insertion, deletion, substitution]   # 优先级
                compare_val = [substitution, insertion, deletion]   # 优先级

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []  # 保存 hyp与ref 中所有对齐的元素下标
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1

            # 出边界后，这里仍然使用，应为第一行与第一列必然是全零的
            i -= 1
            j -= 1
        # elif ops_matrix[i_idx][j_idx] == 1:   # insert
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
        # elif ops_matrix[i_idx][j_idx] == 2:   # delete
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
        # elif ops_matrix[i_idx][j_idx] == 3:   # substitute
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1

        # 出边界处理
        if i < 0 and j >= 0:
            nb_map['D'] += 1
        elif j < 0 and i >= 0:
            nb_map['I'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt

    # print("ref: %s" % " ".join(reference))
    # print("hyp: %s" % " ".join(hypothesis))
    # print(nb_map)
    # print("match_idx: %s" % str(match_idx))
    return wrong_cnt, match_idx, nb_map


def test():
    """
    id: (301225575230191207_spkb_f-301225575230191207_spkb_f_slice19)
    Scores: (#C #S #D #I) 27 4 1 2
    REF:  然 后 而 且 这 个 账 号 ， 你 这 边 *** 做 车 商 续 费 的 话 就 发 真 车 应 该 *** 稍 微 再 便 宜 点 。
    HYP:  然 后 而 且 这 个 账 号 *** 你 这 边 要 做 车 商 续 费 的 话 就 发 真 车 应 该 还 有 一 个 便 宜 的 。
    Eval:
    :return:
    """
    wrong_cnt, match_idx, nb_map = levenshtein_distance(
        reference=list('abcdef'),
        hypothesis=list('cdefg')
    )
    print('test1: ',wrong_cnt, match_idx, nb_map)
    wrong_cnt, match_idx, nb_map = levenshtein_distance(
        reference=list('cdefg'),
        hypothesis=list('abcdef')
    )
    print('test2: ',wrong_cnt, match_idx, nb_map)
    wrong_cnt, match_idx, nb_map = levenshtein_distance(
        reference=list('cdefg'),
        hypothesis=list('')
    )
    print('test3: ',wrong_cnt, match_idx, nb_map)
    wrong_cnt, match_idx, nb_map = levenshtein_distance(
        reference=list(''),
        hypothesis=list('')
    )
    print('test4: ',wrong_cnt, match_idx, nb_map)
    wrong_cnt, match_idx, nb_map = levenshtein_distance(
        reference=list('abcdf'),
        hypothesis=list('bbdef')
    )

    wrong_cnt, match_idx, nb_map = levenshtein_distance(
        hypothesis=list('然后而且这个账号，你这边做车商续费的话就发真车应该稍微再便宜点。'),
        reference=list('然后而且这个账号你这边要做车商续费的话就发真车应该还有一个便宜的。')
    )
    print('test5: ',wrong_cnt, match_idx, nb_map)
if __name__ == '__main__':
    test()
