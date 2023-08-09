from extractor import BNPExtractor
import re
import numpy as np
import evaluate
from scorer.rouge_scorer import RougeScoreAggregated
from rouge_score import scoring

class Evaluator():
    
    def __init__(self) -> None:
        self.extractor = BNPExtractor()
        self.bleu = evaluate.load("bleu")
        self.bertscore = evaluate.load("bertscore")


    def filter_common_bnps(self, bnps):
        common_bnps = ['restaurant', 'aspect', 'typical', 'customer', 'review', 'fact']
        for bnp in common_bnps:
            pattern = re.compile(f',*[^,]*{bnp}[^,]*')
            bnps = re.sub(pattern,'', bnps)
            if bnps and bnps[0] == ',': bnps = bnps[2:]
        # remove stopwords from bnp list
        filtered_bnps = [bnp for bnp in bnps.split(', ') if bnp not in self.extractor.stopwords]
            
        return filtered_bnps
    
    def calculate_metrics_by_bnp(self, preds, labels, filter_bnps_flag=False):
        true_pos = 0
        gold_totals = 0
        pred_totals = 0
        bnp_list = []
        precision, recall, f1 = 0, 0, 0
        for golds, pred in zip(labels, preds):
            #print(f'Labels: {golds} Pred: {preds}')
            #print('BNPs : ', getBNPs(preds))
            if pred:
                pred = pred.strip().lower()
                if filter_bnps_flag:
                    bnps = self.filter_common_bnps(", ".join(self.extractor.extract_from_text(text=pred)))
                    bnp_list.append(", ".join(bnps))
                else:
                    bnps = self.extractor.extract_from_text(text=pred)
                    bnp_list.append(", ".join(bnps))
                gold_pos = set(golds.lower().split(", "))
                gold_totals += len(gold_pos)
                pred_labels = set(bnps)
                pred_totals += len(pred_labels)
                pred = ",".join(pred_labels)
                true_pos += len([True for aspect in gold_pos if aspect in pred])
            else:
                bnp_list.append('')

        print( true_pos, gold_totals, pred_totals)
        if true_pos:
            recall = 100 * (true_pos / gold_totals)
            precision = 100 * (true_pos / pred_totals)
            f1 = (2 * precision * recall) / (precision + recall)
        return (precision, recall, f1), bnp_list

    def calculate_metrics_by_bnp_partial_match(self, preds, labels, filter_bnps_flag=False):
        true_pos = 0
        gold_totals = 0
        pred_totals = 0
        bnp_list = []
        precision, recall, f1 = 0, 0, 0
        for golds, pred in zip(labels, preds):
            #print(f'Labels: {golds} Pred: {preds}')
            #print('BNPs : ', getBNPs(preds))
            if pred:
                pred = pred.strip().lower()
                if filter_bnps_flag:
                    bnps = self.filter_common_bnps(", ".join(self.extractor.extract_from_text(text=pred)))
                    bnp_list.append(", ".join(bnps))
                else:
                    bnps = self.extractor.extract_from_text(text=pred)
                    bnp_list.append(", ".join(bnps))   
                gold_pos = set(golds.lower().split(", "))
                gold_totals += len(gold_pos)
                pred_labels = set(bnps)
                pred_totals += len(pred_labels)
                pred_tokens = set([token.text for token in self.extractor.tokenize(" ".join(pred_labels))])
                for gold in gold_pos:
                    gold_tokens = self.extractor.tokenize(gold)
                    for token in gold_tokens:
                        if token.text in pred_tokens and token.text not in self.extractor.stopwords:
                            true_pos += 1
                            break

        print( true_pos, gold_totals, pred_totals)
        if true_pos:
            recall = 100 * (true_pos / gold_totals)
            precision = 100 * (true_pos / pred_totals)
            f1 = (2 * precision * recall) / (precision + recall)
        return (precision, recall, f1)

    '''
    This method will be used when output generated is a list
    Verify how many true labels matching the items in the predicted list
    e.g.
    "
        1. Allowing pets: Most restaurants do not allow pets, but this one does.
        2. Frozen drinks: Many restaurants do not offer frozen drinks.
        3. Beer on tap: Not all restaurants offer beer on tap.
        4. Stuffed flounder: This is an atypical dish to find on a restaurant menu.
        5. Fried cheesecake: This is an unusual dessert to find in a restaurant."

    '''

    def calculate_metrics_by_list(self, preds, labels):
        gold_totals = 0
        true_pos = 0
        false_pos = 0
        precision, recall, f1 = 0, 0, 0
        for golds, pred in zip(labels, preds):
            if pred:
                pred = pred.strip().lower()
                golds = set(golds.lower().split(", "))
                pred = pred.split('\n')
                matched_predictions = set()
                for gold in golds:
                    for i, prediction in enumerate(pred):
                        if gold in prediction:
                            true_pos += 1
                            matched_predictions.add(i)
                            break
                false_pos += len(pred) - len(matched_predictions)
                gold_totals += len(golds)
            
        print(true_pos, gold_totals, false_pos)
        if true_pos != 0:
            recall = 100 * (true_pos / gold_totals)
            precision = 100 * (true_pos / (true_pos + false_pos))
            f1 = (2 * precision * recall) / (precision + recall)
        
        return (precision, recall, f1)


    def calculate_metrics_by_list_partial_match(self, preds, labels):
        gold_totals = 0
        pred_totals = 0
        true_pos = 0
        false_pos = 0
        precision, recall, f1 = 0, 0, 0
        for golds, pred in zip(labels, preds):
            if pred:
                pred = pred.strip().lower()
                golds = set(golds.lower().split(", "))
                pred = pred.split('\n')
                matched_predictions = set()
                pred_tokens = []
                for prediction in pred:
                    pred_tokens.append([token.text for token in self.extractor.tokenize(prediction)])

                for gold in golds:
                    matched = False
                    gold_tokens = self.extractor.tokenize(gold)
                    for i, prediction in enumerate(pred_tokens):
                        for token in gold_tokens:
                            if token.text in prediction and token.text not in self.extractor.stopwords:
                                matched = True
                                true_pos += 1
                                matched_predictions.add(i)
                                break
                        if matched: break
                false_pos += len(pred) - len(matched_predictions)
                gold_totals += len(golds)
            
        print(true_pos, gold_totals, false_pos)
        if true_pos != 0:
            recall = 100 * (true_pos / gold_totals)
            precision = 100 * (true_pos / (true_pos + false_pos))
            f1 = (2 * precision * recall) / (precision + recall)
        return (precision, recall, f1)



    def calculate_metrics(self, preds, labels):
        true_pos = 0
        gold_totals = 0
        pred_totals = 0
        bnp_list = []
        precision, recall, f1 = 0, 0, 0
        for golds, pred in zip(labels, preds):
            if pred:
                pred = pred.strip().lower()
                bnp_list.append(BNPExtractor().extract_from_text(text=pred))
                gold_pos = set(golds.lower().split(", "))
                gold_totals += len(gold_pos)
                pred_labels = set(bnp_list[-1])
                pred_totals += len(pred_labels)
                pred = ",".join(pred_labels)
                true_pos += len([True for aspect in gold_pos if aspect in pred])

        if true_pos:
            recall = 100 * (true_pos / gold_totals)
            precision = 100 * (true_pos / pred_totals) if pred_totals > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall)
        return {'precision': round(precision, 2),'recall': round(recall, 2), 'f1': round(f1, 2)}

    def format_labels(self, pred, format):
        if format in pred:
            bnp_list = pred.split(format)
            if len(bnp_list) > 1:
                return bnp_list[1].strip().lower()
            else:
                return bnp_list[0].strip().lower()
        else:
            return ''

        
    def calculate_metrics_by_format(self, preds, labels):
        true_pos = 0
        gold_totals = 0
        pred_totals = 0
        bnp_list = []
        precision, recall, f1 = 0, 0, 0
        format = 'Atypical aspects:'
        for golds, pred in zip(labels, preds):
            if pred:
                pred = self.format_labels(pred, format)
                bnp_list.append(BNPExtractor().extract_from_text(text=pred))
                gold_pos = set(golds.lower().split(", "))
                gold_totals += len(gold_pos)
                pred_labels = set(bnp_list[-1])
                pred_totals += len(pred_labels)
                pred = ",".join(pred_labels)
                true_pos += len([True for aspect in gold_pos if aspect in pred])

        if true_pos:
            recall = 100 * (true_pos / gold_totals)
            precision = 100 * (true_pos / pred_totals) if pred_totals > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall)
        return {'precision': round(precision, 2),'recall': round(recall, 2), 'f1': round(f1, 2)}

    def calculate_metrics_exact_match(self, preds, labels):
        print('Evalauting using exact match')
        true_pos = 0
        gold_total = 0
        pred_total = 0
        precision, recall, f1 = 0, 0, 0
        for golds, pred in zip(labels, preds):
            print(f'golds: {golds}, pred: {pred}')
            #if golds:
            golds = set(golds.lower().split(", "))
            #if pred:
            pred = pred.strip().lower().split(", ")
            gold_total += len(golds)
            pred_total += len(pred)
            num_gold_in_pred = len([True for aspect in golds if aspect in pred])
            true_pos += num_gold_in_pred
            print(f'num_gold_in_pred: {num_gold_in_pred}, gold len: {len(golds)}, pred len: {len(pred)}, true pos: {true_pos}')
        if true_pos:
            print(f'true_pos : {true_pos}, gold_total: {gold_total}, pred_total: {pred_total}' )
            recall = 100 * (true_pos / gold_total)
            precision = 100 * (true_pos / pred_total) if pred_total > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall)
        return {'precision': round(precision, 2),'recall': round(recall, 2), 'f1': round(f1, 2)}
    
    
    def getBestMatchedPhrase(self, phrase_1_tokens, phrase_list):
        jaccard_sim_list = []
        for phrase_2_tokens in phrase_list:
            intersection = phrase_1_tokens.intersection(phrase_2_tokens)
            jaccard_sim = len(intersection) / len(phrase_1_tokens.union(phrase_2_tokens))
            jaccard_sim_list.append(jaccard_sim)
        print(jaccard_sim_list)
        if max(jaccard_sim_list) > 0:
            return phrase_list[jaccard_sim_list.index(max(jaccard_sim_list))]
        return ''

    def calculate_metrics_exact_match_with_partial(self, preds, labels, tokenize=True):
        print('Evalauting using exact match with partial score addition: tokenize:',tokenize)
        true_pos_e, true_pos_g = 0, 0
        false_pos = 0
        false_neg = 0
        precision, recall, f1 = 0, 0, 0
        for golds, pred in zip(labels, preds):
            ep_list = []
            gp_list = []
            print(f'golds: {golds}\n pred: {pred}')
            if pred:
                pred = pred.strip().lower().split(", ")
                for ep in pred:
                    if tokenize:
                        ep_list.append(frozenset([token.text.lower() for token in self.extractor.tokenize(ep)]))
                    else:
                        ep_list.append(frozenset([word for word in ep.split(' ')]))
            if golds:
                gold_pos = set(golds.lower().split(", "))
                for gp in gold_pos:
                    if tokenize:
                        gp_list.append(frozenset([token.text.lower() for token in self.extractor.tokenize(gp)]))
                    else:
                        gp_list.append(frozenset([word for word in gp.split(' ')]))
            print(f'ep_list: {ep_list}\n gp_list: {gp_list}')
            if not ep_list and not gp_list:
                true_pos_e += 1
                true_pos_g += 1
            if ep_list and not gp_list:
                false_neg += 1
            if not ep_list and gp_list:
                false_pos += 1

            bestMatchedEP, bestMatchedGP = {}, {}
            for gp_tokens in gp_list:
                if ep_list:
                    bestMatchedEP[gp_tokens] = self.getBestMatchedPhrase(gp_tokens, ep_list)
            for ep_tokens in ep_list:
                if gp_list:
                    bestMatchedGP[ep_tokens] = self.getBestMatchedPhrase(ep_tokens, gp_list)
            for gp_tokens in gp_list:
                print('gp_tokens: ',gp_tokens)
                if ep_list:
                    #ep_tokens = getBestMatchedPhrase(gp_tokens, ep_list)
                    matched_EP = bestMatchedEP[gp_tokens]
                    if matched_EP and bestMatchedGP[matched_EP] == gp_tokens: ep_tokens = matched_EP
                    else: ep_tokens = ''
                    # If there is any partial match
                    if ep_tokens:
                        true_pos_e += len(ep_tokens.intersection(gp_tokens)) / len(ep_tokens)
                        true_pos_g += len(ep_tokens.intersection(gp_tokens)) / len(gp_tokens)
                        false_pos += len(ep_tokens - gp_tokens) / len(ep_tokens)
                        false_neg += len(gp_tokens - ep_tokens) / len(gp_tokens)
                        ep_list.remove(ep_tokens)
                    else:
                        # There is no partial match
                        false_neg += 1
                else:
                    false_neg += 1
                #print(f'true_pos_e: {true_pos_e}, true_pos_g: {true_pos_g}, false_pos: {false_pos}, false_neg: {false_neg}')
            if ep_list:
                # No prediction to match with gold
                false_pos += len(ep_list)
            print(f'true_pos_e: {true_pos_e}, true_pos_g: {true_pos_g}, false_pos: {false_pos}, false_neg: {false_neg}')
        if true_pos_e or true_pos_g:
            precision = 100 * true_pos_e / (true_pos_e + false_pos)
            recall = 100 * true_pos_g / (true_pos_g + false_neg)
            f1 = (2 * precision * recall) / (precision + recall)
        return {'precision': round(precision, 2),'recall': round(recall, 2), 'f1': round(f1, 2)}

    
    # def calculate_metrics_exact_match_with_partial(self, preds, labels, tokenize=True):
    #     print('Evalauting using exact match with partial score addition: tokenize:',tokenize)
    #     true_pos_e, true_pos_g = 0, 0
    #     false_pos = 0
    #     false_neg = 0
    #     precision, recall, f1 = 0, 0, 0
    #     for golds, pred in zip(labels, preds):
    #         ep_list = []
    #         gp_list = []
    #         print(f'golds: {golds}\n pred: {pred}')
    #         if pred:
    #             pred = pred.strip().lower().split(", ")
    #             for ep in pred:
    #                 if tokenize:
    #                     ep_list.append({token.text.lower() for token in self.extractor.tokenize(ep)})
    #                 else:
    #                     ep_list.append({word for word in ep.split(' ')})
    #         if golds:
    #             gold_pos = set(golds.lower().split(", "))
    #             for gp in gold_pos:
    #                 if tokenize:
    #                     gp_list.append({token.text.lower() for token in self.extractor.tokenize(gp)})
    #                 else:
    #                     gp_list.append({word for word in gp.split(' ')})
    #         print(f'ep_list: {ep_list}\n gp_list: {gp_list}')
    #         if not ep_list and not gp_list:
    #             true_pos_e += 1
    #             true_pos_g += 1
    #         if ep_list and not gp_list:
    #             false_pos += 1
    #         if not ep_list and gp_list:
    #             false_neg += 1
    #         for gp_tokens in gp_list:
    #             print('gp_tokens: ',gp_tokens)
    #             if ep_list:
    #                 ep_tokens = self.getBestMatchedPhrase(gp_tokens, ep_list)
    #                 # If there is any partial match
    #                 if ep_tokens:
    #                     true_pos_e += len(ep_tokens.intersection(gp_tokens)) / len(ep_tokens)
    #                     true_pos_g += len(ep_tokens.intersection(gp_tokens)) / len(gp_tokens)
    #                     false_pos += len(ep_tokens - gp_tokens) / len(ep_tokens)
    #                     false_neg += len(gp_tokens - ep_tokens) / len(gp_tokens)
    #                     ep_list.remove(ep_tokens)
    #                 else:
    #                     # There is no partial match
    #                     false_neg += 1
    #             else:
    #                 false_neg += 1
    #             #print(f'true_pos_e: {true_pos_e}, true_pos_g: {true_pos_g}, false_pos: {false_pos}, false_neg: {false_neg}')
    #         if ep_list:
    #             # No prediction to match with gold
    #             false_pos += len(ep_list)
    #         print(f'true_pos_e: {true_pos_e}, true_pos_g: {true_pos_g}, false_pos: {false_pos}, false_neg: {false_neg}')
    #     if true_pos_e or true_pos_g:
    #         precision = 100 * true_pos_e / (true_pos_e + false_pos)
    #         recall = 100 * true_pos_g / (true_pos_g + false_neg)
    #         f1 = (2 * precision * recall) / (precision + recall)
    #     return {'precision': round(precision, 2),'recall': round(recall, 2), 'f1': round(f1, 2)}


    def calculate_metrics_by_clusters(self, preds, labels, train_clusters, aspect_cluster_map):
        print('Evalauting using clusters')
        true_pos = 0
        gold_totals = 0
        new_gold_totals = 0
        new_true_pos = 0
        pred_totals = 0
        precision, recall, f1 = 0, 0, 0
        for golds, pred in zip(labels, preds):
            pred = pred.strip().lower().split(", ")
            golds = set(golds.lower().split(", "))
            gold_totals += len(golds)
            pred_totals += len(pred)
            num_gold_in_pred = len([True for aspect in golds if aspect in pred])
            true_pos += num_gold_in_pred

            new_golds = [gold for gold in golds if aspect_cluster_map[gold] not in train_clusters]
            num_new_gold_in_pred = len([True for aspect in new_golds if aspect in pred])
            new_true_pos += num_new_gold_in_pred
            new_gold_totals += len(new_golds)

            print(f'num_gold_in_pred: {num_gold_in_pred}, gold len: {len(golds)}, pred len: {len(pred)}, true pos: {true_pos}, '\
                 f'num_new_gold_in_pred: {num_new_gold_in_pred}, new_golds len: {len(new_golds)}, new_true_pos: {new_true_pos}')

        if true_pos and new_true_pos:
            recall = 100 * (new_true_pos / new_gold_totals)
            precision = 100 * (true_pos / pred_totals)
            f1 = (2 * precision * recall) / (precision + recall)
        return {'precision': round(precision, 2),'recall': round(recall, 2), 'f1': round(f1, 2)}
    

    def calculate_metrics_by_clusters_abstractive(self, preds, labels, train_clusters, aspect_cluster_map, bnp_to_sent_mapping, golds_to_sentences):
        print('Evalauting using clusters')
        p = []
        r = []
        f1 = []
        new_golds_list = []
        for golds, pred in zip(labels, preds):
            #pred = pred.strip().lower().split(", ")
            #print(golds)
            golds = golds.strip()
            sents = golds_to_sentences[golds_to_sentences['abs_true_nc'] == golds]['sentences'].tolist()
            if not sents:
                print(f'Match not found for:#{golds}#')
                golds = None
            else:
                golds = sents[0].split('|')
            #golds = set(self.extractor.extract_sentences(golds))
            new_golds = set()
            for gold in golds:
                gold = gold.strip()
                for bnp in bnp_to_sent_mapping[bnp_to_sent_mapping['abstractive'] == gold]['extractive'].tolist():
                    if aspect_cluster_map[bnp.strip().lower()] not in train_clusters: new_golds.add(gold)

            new_golds = " ".join(new_golds)
            new_golds_list.append(new_golds)
            if new_golds:
                #print('new_golds')
                results = self.bertscore.compute(predictions=[pred], references=[new_golds], model_type="microsoft/deberta-large-mnli")
                p.append(results['precision'])
                r.append(results['recall'])
                f1.append(results['f1'])
        if p and r and f1:
            return {'precision': round(np.mean(p), 2),'recall': round(np.mean(r), 2), 'f1': round(np.mean(f1), 2), 'new_golds': new_golds_list}
        else:
            return {'precision': round(0, 2),'recall': round(0, 2), 'f1': round(0, 2), 'new_golds': new_golds_list}

    
    def compute_bleu(self, preds, labels):
        bleu_score = 0
        for pred, gold in zip(preds, labels):
            print(f'gold: {gold}\n pred: {pred}')
            results = self.bleu.compute(predictions=pred, references=gold)
            print(results)
            bleu_score += results['bleu']
        return {'bleu': round(bleu_score / len(gold), 2), 'total_bleu': round(bleu_score, 2)}


    def compute_rouge_aggregated(self, predictions, references, rouge_types=None,
                                  use_aggregator=True, use_stemmer=False, tokenizer=None):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"]

        scorer = RougeScoreAggregated(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer)
        scores_ngrams, result = {}, {}
        for rouge_type in rouge_types:
            scores_ngrams[rouge_type] = {'matched_ngram_count': 0, 'target_ngram_count': 0, 'prediction_ngrams_count': 0}
        if use_aggregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores_lcs = []

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            for rouge_type in ["rouge1", "rouge2", "rouge3", "rouge4"]:
                scores_ngrams[rouge_type]['matched_ngram_count'] += score[rouge_type]['matched_ngram_count']
                scores_ngrams[rouge_type]['target_ngram_count'] += score[rouge_type]['target_ngram_count']
                scores_ngrams[rouge_type]['prediction_ngrams_count'] += score[rouge_type]['prediction_ngrams_count']
            score_l = {key: score[key] for key in ["rougeL", "rougeLsum"]}
            if score_l:
                if use_aggregator:
                    aggregator.add_scores(score_l)
                else:
                    scores_lcs.append(score_l)
        
        for rouge_type in ["rouge1", "rouge2", "rouge3", "rouge4"]:
            precision = scores_ngrams[rouge_type]['matched_ngram_count'] / max(scores_ngrams[rouge_type]['prediction_ngrams_count'], 1)
            recall = scores_ngrams[rouge_type]['matched_ngram_count'] / max(scores_ngrams[rouge_type]['target_ngram_count'], 1)
            result[rouge_type] = {'precision': round(precision, 2), 'recall': round(recall, 2), 'f1': round(scoring.fmeasure(precision, recall), 2)}

        if use_aggregator:
            aggregated_result = aggregator.aggregate()
            for key in aggregated_result:
                print(f'{key}: P = {round(aggregated_result[key].mid.precision, 2)}, R = {round(aggregated_result[key].mid.recall, 2)}, F1 = {round(aggregated_result[key].mid.fmeasure, 2)}')
                result[key] = aggregated_result[key].mid.fmeasure

        else:
            #result = {}
            for key in scores_lcs[0]:
                plist = []
                rlist = []
                flist = []
                for score in scores_lcs:
                    plist.append(round(score[key].precision, 2))
                    rlist.append(round(score[key].recall, 2))
                    flist.append(round(score[key].fmeasure, 2))
                print(f'{key}: P = {plist}, R = {rlist}, F1 = {flist}')
                result[key] = list(score[key].fmeasure for score in scores_lcs)
            
        return result
    

    def compute_bertscore(self, predictions, references, model_type="microsoft/deberta-large-mnli"):
        preds = []
        refs = []
        precision, recall, f1 = [], [], []
        for prediction, reference in zip(predictions, references):
            if reference == '' and prediction == '':
                precision.append(1)
                recall.append(1)
                f1.append(1)
            else:
                preds.append(prediction)
                refs.append(reference)
        
        result = self.bertscore.compute(predictions=preds, references=refs, model_type=model_type)
        precision.extend(result['precision'])
        recall.extend(result['recall'])
        f1.extend(result['f1'])
        
        return {'precision': precision, 'recall': recall, 'f1': f1}


    def compute_clasification_metrics(self, preds, labels):
        true_pos, true_neg, false_neg, false_pos = 0, 0, 0, 0

        for gold, pred in zip(preds, labels):
            if gold == '':
                if gold == pred:
                    true_neg += 1
                else:
                    false_pos += 1
            else:
                if pred == '':
                    false_neg += 1
                else:
                    true_pos += 1

        print(true_neg, true_pos, false_neg, false_pos)
        return {'core_recall': round(true_neg / max(true_neg + false_pos, 1), 2), 'noncore_neg_recall': round(false_neg / max(true_pos + false_neg, 1), 2)}
                    