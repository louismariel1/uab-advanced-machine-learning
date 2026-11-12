# Advanced Machine Learning — Academic Project Proposals

**Status:** Proposed projects — awaiting professor feedback and selection. No experimental results are claimed.

This repository presents two alternative research-oriented projects. Each evaluates a **multi-stage learning pipeline** against a controlled reference pipeline, rather than treating learning methods as unrelated laboratory exercises.

## At a glance

| | Project A | Project B |
|---|---|---|
| **Title** | Efficient Image Classification through SSL and KD: Dense Fine-Tuning versus LoRA | Mitigating Catastrophic Forgetting through Transfer Learning and KD |
| **Research challenge** | Accuracy versus teacher-adaptation efficiency | Learning new classes without forgetting old classes |
| **Reference pipeline** | SSL → Dense Fine-Tuning → KD | Transfer Learning → Sequential Fine-Tuning |
| **Proposed pipeline** | SSL → LoRA Fine-Tuning → KD | Transfer Learning → Continual Learning with KD |
| **Dataset** | Unlabeled CIFAR-10 source; five-class CIFAR-100 target | CIFAR-100, 20 classes across four incremental tasks |
| **Primary outcomes** | Final student test accuracy; LoRA adaptation efficiency | Final average accuracy; catastrophic forgetting |
| **Expected contribution** | Determine whether LoRA retains dense-pipeline accuracy with lower adaptation cost | Determine whether distillation reduces forgetting during incremental learning |
| **Implementation readiness** | Higher: directly informed by three practical labs | Requires additional continual-learning implementation |

## Project A — Summary

**Research question:** Can SSL → LoRA → KD achieve final compact-student accuracy comparable to SSL → Dense Fine-Tuning → KD while reducing teacher-adaptation resources?

Both pipelines share the same SSL-pretrained teacher checkpoint, target data, student architecture, and KD procedure. Only the teacher-adaptation method differs. The main hypothesis is **accuracy non-inferiority** (proposed margin: 2 percentage points), combined with fewer trainable adaptation parameters. Accuracy improvement is exploratory, not assumed.

**[Read full Project A proposal](PROJECT_A.md)**

## Project B — Summary

**Research question:** Does adding knowledge distillation to transfer-learning-based sequential image classification improve accuracy across learned tasks and reduce catastrophic forgetting?

The reference pipeline sequentially fine-tunes a transferred model on new classes. The proposed pipeline additionally distills predictions from the frozen previous model at each task transition. Both use the same pretrained initialization, task sequence, and architecture. The main outcomes are **final average accuracy** and **average forgetting** in class-incremental evaluation.

**[Read full Project B proposal](PROJECT_B.md)**

## Selection considerations

- **Choose Project A** for a bounded accuracy–efficiency study closely aligned with the supplied SSL, LoRA, and KD labs.
- **Choose Project B** for a focused study of the stability–plasticity problem in continual learning, with additional implementation complexity.

**Selection requested:** Professor feedback on scientific relevance, scope, computational feasibility, and preferred project. Final datasets, hyperparameters, and deliverables remain subject to course requirements and approval.
