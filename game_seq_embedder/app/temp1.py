import torch
import ipdb
import bibtexparser


def target_distribution(qij: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float

    qij shape: point number x centre number
    $$
    p_{i j}=\frac{q_{i j}^{2} / f_{j}}{\sum_{j^{\prime}} q_{i j^{\prime}}^{2} / f_{j^{\prime}}}
    $$

    """
    weight = (qij ** 2) / torch.sum(qij, dim=0)
    return (weight.t() / torch.sum(weight, 1)).t()


def main():
    t1 = torch.randn(6, 7)
    target_distribution(t1)


class BibtextReader:

    def __init__(self):
        self.bibtexts = {}

    def read_bibtext_from_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            # Load all bibtext from txt file
            bibtext_entries = []
            bibtext_entry = ''
            for line in f:
                if not line.strip():
                    bibtext_entry = bibtext_entry.replace('```', '')
                    bibtext_entries.append(bibtexparser.loads(bibtext_entry).entries[0])
                    bibtext_entry = ''
                else:
                    bibtext_entry += line.strip()

        for entry in bibtext_entries:
            self.bibtexts[entry['title']] = entry


if __name__ == '__main__':
    bibtext_path = '/Users/jiashupu/Documents/writings_on_NLP_ML/Paper/Game bert/literature_review/Bibtex.md'
    bib_reader = BibtextReader()
    bib_reader.read_bibtext_from_txt(bibtext_path)

