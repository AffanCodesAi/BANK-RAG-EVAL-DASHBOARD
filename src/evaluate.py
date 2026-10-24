import re
import sys
import time

import chromadb
from groq import Groq

from search import (
    CHROMA_PATH,
    COLLECTION_NAME,
    GROQ_MODEL,
    ROOT,
    build_context,
    generate_answer,
    load_embedding_model,
    retrieve_documents,
)


TESTS = [
    {
        "name": "Raast transaction fee",
        "question": "What is the fee for a Raast transaction?",
        "expected": ["charge"],
        "should_refuse": False,
    },
    {
        "name": "Raast per-transaction limit",
        "question": "What is the Raast limit per transaction?",
        "expected": ["500,000", "500000"],
        "should_refuse": False,
    },
    {
        "name": "Raast daily limit",
        "question": "What is the daily Raast limit?",
        "expected": ["1,000,000", "1000000"],
        "should_refuse": False,
    },
    {
        "name": "Unsupported special business limit",
        "question": (
            "What is the special Raast daily limit for a business "
            "account under a corporate package?"
        ),
        "expected": [],
        "should_refuse": True,
    },
]

REFUSAL_PHRASES = [
    "could not confirm",
    "cannot confirm",
    "do not specify",
    "does not specify",
    "not enough information",
    "cannot verify",
    "unable to confirm",
    "not provided",
    "not available in the documents",
    "could not find any information",
    "do not contain any information",
    "does not contain any information",
    "do not mention",
    "does not mention",
    "supplied documents",
    "provided documents",
    "contact js bank",
]


def main():
    import os

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("GROQ_API_KEY is missing. Check your .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key, timeout=30, max_retries=0)

    try:
        db = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = db.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        print(f"Could not open ChromaDB: {exc}")
        sys.exit(1)

    if collection.count() == 0:
        print("The knowledge base is empty. Run ingestion first.")
        sys.exit(1)

    print("Loading embedding model...")
    embedding_model, device = load_embedding_model()
    print(f"Answer model: {GROQ_MODEL}")
    print(f"Embedding device: {device}")
    print(f"Indexed chunks: {collection.count()}")
    print(f"Evaluation cases: {len(TESTS)}")
    print("=" * 65)

    passed = 0

    for index, test in enumerate(TESTS, start=1):
        if index > 1:
            print("Waiting 15 seconds to reduce rate-limit pressure...")
            time.sleep(15)
        print(f"\n[{index}/{len(TESTS)}] {test['name']}")
        print("Question:", test["question"])

        start = time.perf_counter()

        try:
            retrieved = retrieve_documents(
                test["question"],
                embedding_model,
                collection,
            )

            context, sources = build_context(retrieved)
            context_lower = context.lower()
            expected_in_context = (
                test['should_refuse']
                or any(term.lower() in context_lower for term in test['expected'])
            )
            if not test['should_refuse']:
                print(
                    'Expected information in retrieved context:',
                    'PASS' if expected_in_context else 'FAIL — retrieval miss',
                )

            if not context.strip():
                print("FAIL: No usable context retrieved.")
                continue

            answer = generate_answer(test["question"], context, client)
            elapsed = time.perf_counter() - start

            print("Answer:", answer)
            print("Response time:", f"{elapsed:.2f} seconds")
            print("Sources retrieved:", len(sources))

            answer_lower = answer.lower()
            citation_found = bool(re.search(r"(?:\[|【)\s*Source\s+\d+\s*(?:\]|】)", answer, re.IGNORECASE))

            if test["should_refuse"]:
                answer_check = any(
                    phrase in answer_lower for phrase in REFUSAL_PHRASES
                )
                print(
                    "Refusal check:",
                    "PASS" if answer_check else "FAIL",
                )
            else:
                answer_check = any(
                    term.lower() in answer_lower
                    for term in test["expected"]
                )
                print(
                    "Expected information check:",
                    "PASS" if answer_check else "FAIL",
                )

            citation_required = not (test["should_refuse"] and answer_check)
            citation_ok = citation_found or not citation_required
            print(
                "Citation check:",
                "PASS" if citation_found else (
                    "PASS (not required for refusal)"
                    if not citation_required else "FAIL"
                ),
            )

            retrieval_ok = test["should_refuse"] or expected_in_context
            case_passed = answer_check and citation_ok and retrieval_ok
            print("Overall:", "PASS" if case_passed else "REVIEW")
            if case_passed:
                passed += 1

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 65)
    print(f"Evaluation summary: {passed}/{len(TESTS)} cases passed")
    print("Review every answer manually; keyword checks are not proof of accuracy.")


if __name__ == "__main__":
    main()
