"""
ChatGPT contribution note:
Docstrings added to functions and minor tweaks to appearance
"""

import argparse
from typing import Set, Dict, Tuple, List
import pandas as pd


def read_mgi_file(file_path: str) -> pd.DataFrame:
    """
    Parse an MGI gene-association (GAF) file.

    * Skips comment lines beginning with '!'
    * Uses the fragile, file-embedded header comment block (lines '!\\t1. …')
      to label the 17 standard GAF columns.
    * Returns a string-typed DataFrame with one row per annotation.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the `gene_association.mgi` file.

    Returns
    -------
    pd.DataFrame
        A DataFrame whose columns match the header lines in the file.
    """

    def read_headers(file_path: str) -> List[str]:
        """
        Extract GAF column names from the leading comment block.

        The function expects comment lines exactly in the form
        '!\\t1. DB', '!\\t2. DB Object ID', … and stops at the first
        data row. If the format ever changes, header parsing fails.

        Parameters
        ----------
        file_path : str
            Path to the same file being opened in the outer scope.

        Returns
        -------
        list[str]
            The ordered list of header strings.
        """

        headers = []
        index = 1
        with open(file_path) as fp:
            for line in fp:
                if not line.startswith("!"):
                    break
                prefix = f"!\t{index}.".ljust(6, " ")
                if line.startswith(prefix):
                    # The optional description of the header is after two white spaces which is not used
                    # If there is no optional description, we need to remove the line end character "\n"
                    header = line[len(prefix) :].split("  ")[0].rstrip("\n")
                    headers.append(header)
                    index += 1
        return headers

    headers = read_headers(file_path)

    df_mgi = pd.read_csv(file_path, sep="\t", comment="!", header=None, names=headers, dtype=str)
    return df_mgi


def propagate_annotations_to_dictionary(
    data: pd.DataFrame, parent_relationship_dictionary: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """
    Given raw annotations and a GO-parent map, produce each gene's full GO set.

    For every gene (DB Object ID) in *data*:
    ─────────────────────────────────────────────
      • start with the direct GO IDs in the row
      • walk up the parent graph (transitively) and
        attach every ancestor to that gene
      • ensure each GO term appears **only once** per gene

    A defensive deep-copy of *parent_relationship_dictionary* is made so
    the caller's structure is not mutated.

    Parameters
    ----------
    data : pd.DataFrame
        GAF rows; must contain columns ``DB Object ID`` and ``GO ID``.
    parent_relationship_dictionary : dict[str, set[str]]
        Mapping child-GO → set{direct-parent-GO} built from *mergeGO.out*.

    Returns
    -------
    dict[str, set[str]]
        gene → set(all GO terms, including propagated parents)
    """
    parent_relationship_dictionary = {
        go: set(parents) for go, parents in parent_relationship_dictionary.items()
    }

    db_object_id_grouped_genome_data = data.groupby("DB Object ID")

    def depth_first_search(go_id: str, parent_go_id_dict: dict[str, Set[str]], checked_go_ids: set):
        """
        Depth-first helper that fills ``parent_go_id_dict[go_id]`` with the term’s ancestors.
        Note: The ```parent_go_id_dict`` is MUTATED IN-PLACE, meaning that it changed during every
        call of this function.

        Parameters
        ----------
        go_id : str
            Current GO term being expanded.
        parent_go_id_dict : dict[str, set[str]]
            The mutable parent-map whose sets grow in place.
        checked_go_ids : set[str]
            A global set of GO terms already visited and not to be visited again.
        """
        if go_id in checked_go_ids:
            return
        checked_go_ids.add(go_id)
        parent_go_ids = set(parent_go_id_dict.get(go_id, []))
        for parent_go_id in parent_go_ids:
            if parent_go_id in parent_go_id_dict:
                if parent_go_id not in checked_go_ids:
                    depth_first_search(parent_go_id, parent_go_id_dict, checked_go_ids)
                parent_go_id_dict[go_id].update(parent_go_id_dict[parent_go_id])

    checked = set()
    genome_class_parents = {}
    for grp_name, grp_df in db_object_id_grouped_genome_data:
        genome_classes = set(grp_df["GO ID"])
        if grp_name not in genome_class_parents:
            genome_class_parents[grp_name] = genome_classes.copy()
        for go_id in genome_classes:
            depth_first_search(go_id, parent_relationship_dictionary, checked)

            genome_class_parents[grp_name].update(parent_relationship_dictionary.get(go_id, []))

    return genome_class_parents


def get_gene_members_per_go_class(gene_to_parent: Dict[str, Set[str]]) -> Dict[str, int]:
    """
    Convert gene→terms into term→unique-gene-count.

    Parameters
    ----------
    gene_to_parent : dict[str, set[str]]
        Output of function ``propagate_annotations_to_dictionary``.

    Returns
    -------
    dict[str, int]
        GO term → number of unique genes annotated to it.
    """
    counts = {}
    for genome, genome_parents in gene_to_parent.items():
        for genome_parent in genome_parents:
            if genome_parent not in counts:
                counts[genome_parent] = 0
            counts[genome_parent] += 1
    return counts


def read_mapping_file(file_path: str) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """
    Parse the mapping file into two look-ups:

    1. GO ID → set{direct parent GO IDs}
    2. GO ID → human-readable GO class name

    Column indices are zero-based:
    • col-2 = GO ID
    • col-3 = GO class label
    • col-4 = comma-separated parent list

    Parameters
    ----------
    file_path : str
        Path to ``mergeGO.out``.

    Returns
    -------
    (dict[str, set[str]], dict[str, str])
        (parent_map, class_name_map)
    """

    merge_out = pd.read_csv(file_path, sep="\t", header=None, dtype=str)
    merge_out.rename(columns={2: "GO ID", 3: "GO Class", 4: "parent GO"}, inplace=True)

    parent_go_id_dict = {
        go_id: set(str(parents).split(",")) if not pd.isnull(parents) else set()
        for go_id, parents in zip(merge_out["GO ID"], merge_out["parent GO"])
    }

    class_names_dict = {
        go_id: class_name for go_id, class_name in zip(merge_out["GO ID"], merge_out["GO Class"])
    }

    return parent_go_id_dict, class_names_dict


def get_top_gene_classes(gene_file_path: str, mapping_file_path: str) -> pd.DataFrame:
    """
    Finds the top 50 GO classes with the highest unique gene counts based on the input files.

    End-to-end pipeline: read both files, propagate, count, and rank.

    Parameters
    ----------
    gene_file_path : str
        Path to `gene_association.mgi`.
    mapping_file_path : str
        Path to `mergeGO.out`.

    Returns
    -------
    pd.DataFrame
        Top-50 GO classes with columns ``["GO ID", "Gene Count", "GO Class"]``.
    """

    # Part 1: Read gene-GO annotations
    gene_go_df = read_mgi_file(gene_file_path)

    # Strip the gene_go_df GO ID column of the "GO:" prefix
    gene_go_df["GO ID"] = gene_go_df["GO ID"].apply(lambda x: x[3:])

    # Part 2 & 3
    parent_go_id_dict, class_names_dict = read_mapping_file(mapping_file_path)

    # Part 4: Propagate annotations
    genome_class_parents = propagate_annotations_to_dictionary(gene_go_df, parent_go_id_dict)

    # Part 5: Count gene members per GO class
    combined_sums = get_gene_members_per_go_class(genome_class_parents)

    # Sort top genes
    top_genes = sorted(combined_sums.items(), key=lambda x: x[1], reverse=True)[:50]

    # Construct and return a dataframe of the top genes
    annotated_top_genes = [(go_id, count, class_names_dict[go_id]) for go_id, count in top_genes]

    top_genes_df = pd.DataFrame(annotated_top_genes, columns=["GO ID", "Gene Count", "GO Class"])

    return top_genes_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print the 50 GO classes with the highest unique gene counts."
    )
    parser.add_argument("gene_association", help="gene_association.mgi file")
    parser.add_argument("merge_go", help="mergesGO.out file")
    args = parser.parse_args()

    top_df = get_top_gene_classes(args.gene_association, args.merge_go)
    print(top_df.to_string(index=False))
