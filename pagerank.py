import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Initialise dictionary with null values
    prob_dist = {entry: 0 for entry in corpus}

    if len(corpus[page]) == 0:
        for entry in prob_dist:
            prob_dist[entry] = 1 / len(corpus)
        return prob_dist

    for entry in prob_dist:
        prob_dist[entry] += (1 - damping_factor) / len(corpus)
        if entry in corpus[page]:
            prob_dist[entry] += damping_factor / len(corpus[page])

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank valurandinte (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    seen = {entry: 0 for entry in corpus}

    # First sample is generated randomly:
    current_page = random.choice(list(seen))
    seen[current_page] += 1

    # Remaining sample are calculated based on transition model
    for _ in range(0, n - 1):

        model = transition_model(corpus, current_page, damping_factor)

        # Pick next page based on the transition model
        total = 0
        random_value = random.random()
        for page, probability in model.items():
            total += probability
            if random_value <= total:
                current_page = page
                break

        seen[current_page] += 1
    # Normalise
    page_ranks = {page: (count / n) for page, count in seen.items()}

    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    current_ranks = {entry: (1 / len(corpus)) for entry in corpus}
    new_ranks = {entry: None for entry in corpus}

    while True:
        for page in corpus:
            acc = 0
            for link in corpus:
                # If page has links
                if page in corpus[link]:
                    acc += current_ranks[link] / len(corpus[link])
                # If page has no links
                if len(corpus[link]) == 0:
                    acc += (current_ranks[link]) / len(corpus)
            acc *= damping_factor
            acc += (1 - damping_factor) / len(corpus)

            new_ranks[page] = acc

        difference = max([abs(new_ranks[_] - current_ranks[_]) for _ in current_ranks])
        if difference < 0.001:
            break
        else:
            current_ranks = new_ranks.copy()

    return current_ranks


if __name__ == "__main__":
    main()
