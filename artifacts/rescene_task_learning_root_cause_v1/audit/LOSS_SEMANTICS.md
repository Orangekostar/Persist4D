# ReScene Loss Semantics

Status: `PASS`

The fixed real batch uses a seed-45 randomly initialized task decoder and the verified pretrained Concerto encoder. The public-code objective includes every returned value once; the local weighted objective excludes per-layer contrastive diagnostics and applies the criterion weight dictionary.

| loss key | class | raw | upstream multiplier | local multiplier | upstream contribution | local contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| loss_aux_contrastive | aggregate | 0 | 1 | 1 | 0 | 0 |
| loss_ce | objective | 3.90063024 | 1 | 2 | 3.90063024 | 7.80126047 |
| loss_ce_0 | objective | 2.9881146 | 1 | 2 | 2.9881146 | 5.97622919 |
| loss_ce_1 | objective | 3.39521718 | 1 | 2 | 3.39521718 | 6.79043436 |
| loss_ce_10 | objective | 4.0454731 | 1 | 2 | 4.0454731 | 8.0909462 |
| loss_ce_11 | objective | 3.25562906 | 1 | 2 | 3.25562906 | 6.51125813 |
| loss_ce_2 | objective | 3.27645969 | 1 | 2 | 3.27645969 | 6.55291939 |
| loss_ce_3 | objective | 3.39390683 | 1 | 2 | 3.39390683 | 6.78781366 |
| loss_ce_4 | objective | 3.54949808 | 1 | 2 | 3.54949808 | 7.09899616 |
| loss_ce_5 | objective | 3.47927928 | 1 | 2 | 3.47927928 | 6.95855856 |
| loss_ce_6 | objective | 3.78283787 | 1 | 2 | 3.78283787 | 7.56567574 |
| loss_ce_7 | objective | 3.23647594 | 1 | 2 | 3.23647594 | 6.47295189 |
| loss_ce_8 | objective | 3.94764423 | 1 | 2 | 3.94764423 | 7.89528847 |
| loss_ce_9 | objective | 3.7667582 | 1 | 2 | 3.7667582 | 7.53351641 |
| loss_dice | objective | 0.819973826 | 1 | 2 | 0.819973826 | 1.63994765 |
| loss_dice_0 | objective | 0.541374803 | 1 | 2 | 0.541374803 | 1.08274961 |
| loss_dice_1 | objective | 0.54563427 | 1 | 2 | 0.54563427 | 1.09126854 |
| loss_dice_10 | objective | 0.596835256 | 1 | 2 | 0.596835256 | 1.19367051 |
| loss_dice_11 | objective | 0.671358883 | 1 | 2 | 0.671358883 | 1.34271777 |
| loss_dice_2 | objective | 0.569188714 | 1 | 2 | 0.569188714 | 1.13837743 |
| loss_dice_3 | objective | 0.824616909 | 1 | 2 | 0.824616909 | 1.64923382 |
| loss_dice_4 | objective | 0.891883492 | 1 | 2 | 0.891883492 | 1.78376698 |
| loss_dice_5 | objective | 0.767242312 | 1 | 2 | 0.767242312 | 1.53448462 |
| loss_dice_6 | objective | 0.591132402 | 1 | 2 | 0.591132402 | 1.1822648 |
| loss_dice_7 | objective | 0.714854538 | 1 | 2 | 0.714854538 | 1.42970908 |
| loss_dice_8 | objective | 0.813047528 | 1 | 2 | 0.813047528 | 1.62609506 |
| loss_dice_9 | objective | 0.617982626 | 1 | 2 | 0.617982626 | 1.23596525 |
| loss_mask | objective | 1.51104522 | 1 | 5 | 1.51104522 | 7.55522609 |
| loss_mask_0 | objective | 0.746848285 | 1 | 5 | 0.746848285 | 3.73424143 |
| loss_mask_1 | objective | 0.978832006 | 1 | 5 | 0.978832006 | 4.89416003 |
| loss_mask_10 | objective | 0.975450516 | 1 | 5 | 0.975450516 | 4.87725258 |
| loss_mask_11 | objective | 1.03078163 | 1 | 5 | 1.03078163 | 5.15390813 |
| loss_mask_2 | objective | 0.968938947 | 1 | 5 | 0.968938947 | 4.84469473 |
| loss_mask_3 | objective | 1.46242571 | 1 | 5 | 1.46242571 | 7.31212854 |
| loss_mask_4 | objective | 1.72623026 | 1 | 5 | 1.72623026 | 8.63115132 |
| loss_mask_5 | objective | 1.22902393 | 1 | 5 | 1.22902393 | 6.14511967 |
| loss_mask_6 | objective | 0.936722755 | 1 | 5 | 0.936722755 | 4.68361378 |
| loss_mask_7 | objective | 1.13870704 | 1 | 5 | 1.13870704 | 5.69353521 |
| loss_mask_8 | objective | 1.34309053 | 1 | 5 | 1.34309053 | 6.71545267 |
| loss_mask_9 | objective | 0.927284002 | 1 | 5 | 0.927284002 | 4.63642001 |
| loss_segment_contrastive | aggregate | 0.693093598 | 1 | 1 | 0.693093598 | 0.693093598 |
| loss_segment_contrastive_layer0 | diagnostic | 0.693093598 | 1 | 0 | 0.693093598 | 0 |

Weighted objective: `185.536102295`.
Raw released-code objective: `71.3446121216`.

## EOS

EOS 0.1 versus 0.2 class-head gradient cosine is `0.998021245003` and relative norm difference is `0.0888135356756`. R5 authorization is `false` under the preregistered cosine 0.98 / relative-norm 0.10 gate.
