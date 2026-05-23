"""Benchmark extractive-QA probe candidates on CPU vs GPU.

Uses spans from real reject samples in run.log so the timing reflects
actual probe workload (200-500 char contexts, SQuAD2-style unanswerable).
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Spans pulled from the user's hotpot-qa run.log reject samples.
CASES: list[dict[str, str | bool]] = [
    {
        "name": "kevin_lima_span0_should_be_unanswerable",
        "question": "Who directed the sequel to the 1996 live-action remake of the Disney animated film based on Dodie Smith's novel?",
        "context": "101 Dalmatians is a 1996 American live-action comedy adventure film based on Walt Disney's animated 1961 movie adaptation of Dodie Smith's 1956 novel \"The Hundred and One Dalmatians.\"",
        "expect_unanswerable": True,
    },
    {
        "name": "kevin_lima_span1_should_extract",
        "question": "Who directed the sequel to the 1996 live-action remake of the Disney animated film based on Dodie Smith's novel?",
        "context": "102 Dalmatians is a 2000 American live action and CG-animated film adventure drama film directed by Kevin Lima in his live-action directorial debut and produced by Edward S. Feldman and Walt Disney Pictures.",
        "expect_unanswerable": False,
    },
    {
        "name": "bad_religion_span0_should_extract",
        "question": "Who was the original drummer in the American punk rock band formed in Los Angeles in 1980 whose discography comprises sixteen studio albums?",
        "context": "The discography of Bad Religion, an American punk rock band, consists of 16 studio albums, two live albums, four compilation albums, one box set, two extended plays (EPs), 29 singles, five video albums and 23 music videos. Formed in Los Angeles, California in 1980, the band originally featured vocalist Greg Graffin, guitarist Brett Gurewitz, bassist Jay Bentley and drummer Jay Ziskrout, who released their self-titled debut EP in February 1981 on Gurewitz's label Epitaph Records.",
        "expect_unanswerable": False,
    },
    {
        "name": "bad_religion_span1_should_also_extract",
        "question": "Who was the original drummer in the American punk rock band formed in Los Angeles in 1980 whose discography comprises sixteen studio albums?",
        "context": "Bad Religion is an American punk rock band from Los Angeles, California. Formed in 1980, the group originally included vocalist Greg Graffin, guitarist Brett Gurewitz, bassist Jay Bentley and drummer Jay Ziskrout.",
        "expect_unanswerable": False,
    },
    {
        "name": "dionysius_span0_should_be_unanswerable",
        "question": "Who devised the calendar era that became the prevalent method in Europe for naming the year later designated 490 BC?",
        "context": "The denomination 490 BC for this year has been used since the early medieval period, when the Anno Domini calendar era became the prevalent method in Europe for naming years.",
        "expect_unanswerable": True,
    },
    {
        "name": "dionysius_span1_should_extract",
        "question": "Who devised the calendar era that became the prevalent method in Europe for naming the year later designated 490 BC?",
        "context": 'The "Anno Domini" dating system was devised in AD 525 by Dionysius Exiguus.',
        "expect_unanswerable": False,
    },
    {
        "name": "dog_span0_should_be_unanswerable",
        "question": "According to the archaeological record, how many years ago were the first undisputed remains of the species identified as the most widely abundant carnivore buried beside humans?",
        "context": 'The domestic dog ("Canis lupus familiaris" or "Canis familiaris") is a member of genus "Canis" (canines) that forms part of the wolf-like canids, and is the most widely abundant carnivore.',
        "expect_unanswerable": True,
    },
    {
        "name": "dog_span1_should_extract",
        "question": "According to the archaeological record, how many years ago were the first undisputed remains of the species identified as the most widely abundant carnivore buried beside humans?",
        "context": "The archaeological record shows the first undisputed dog remains buried beside humans 14,700 years ago, with disputed remains occurring 36,000 years ago.",
        "expect_unanswerable": False,
    },
]


def bench(model_name: str, device: int, n_warmup: int = 3, n_reps: int = 5) -> dict:
    """Return per-case latency and correctness for ``model_name`` on ``device``.

    ``device=-1`` is CPU; otherwise the GPU index.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=device,
        handle_impossible_answer=True,
    )

    # warmup so we don't include compile / cache-prime time in the timing.
    for _ in range(n_warmup):
        for case in CASES[:2]:
            qa(question=case["question"], context=case["context"])

    correct = 0
    per_case_latency_ms: list[float] = []
    answers: list[tuple[str, str, bool]] = []

    for case in CASES:
        t0 = time.perf_counter()
        for _ in range(n_reps):
            out = qa(question=case["question"], context=case["context"])
        elapsed = (time.perf_counter() - t0) / n_reps * 1000.0
        per_case_latency_ms.append(elapsed)

        # SQuAD2 pipelines return answer == "" (and high no-answer score) when
        # impossible — we treat empty answer as the model's "INSUFFICIENT".
        predicted_unanswerable = not out["answer"].strip()
        ok = predicted_unanswerable == case["expect_unanswerable"]
        if ok:
            correct += 1
        answers.append((case["name"], out["answer"][:80], ok))

    return {
        "model": model_name,
        "device": "GPU" if device >= 0 else "CPU",
        "mean_ms": sum(per_case_latency_ms) / len(per_case_latency_ms),
        "p95_ms": sorted(per_case_latency_ms)[int(0.95 * len(per_case_latency_ms))],
        "throughput_qps": 1000.0 / (sum(per_case_latency_ms) / len(per_case_latency_ms)),
        "accuracy": correct / len(CASES),
        "answers": answers,
    }


def main() -> None:
    models = [
        "deepset/deberta-v3-base-squad2",
        "deepset/roberta-base-squad2",
        "deepset/tinyroberta-squad2",
    ]
    print(
        f"CUDA available: {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}"
    )
    print()
    for model_name in models:
        for device in (-1, 0) if torch.cuda.is_available() else (-1,):
            result = bench(model_name, device=device)
            print(f"=== {result['model']}  [{result['device']}] ===")
            print(
                f"  mean: {result['mean_ms']:.1f} ms  p95: {result['p95_ms']:.1f} ms  throughput: {result['throughput_qps']:.1f} q/s"
            )
            print(f"  accuracy on 8 hand-labeled cases: {result['accuracy']:.0%}")
            for name, answer, ok in result["answers"]:
                mark = "OK " if ok else "WRONG"
                print(f"    [{mark}] {name}: {answer!r}")
            print()


if __name__ == "__main__":
    main()
