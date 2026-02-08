#!/usr/bin/env python3
"""
Pipeline: Dock ligands to a prepared receptor, extract distances,
and save protein-ligand complexes.
"""


import subprocess
from pathlib import Path
from vina import Vina
from rdkit import Chem
from rdkit.Chem import AllChem




# ========= USER INPUT =========
protein_pdbqt = "/Users/misahikawa/Documents/clean_4YKN.pdbqt"
protein_pdb = "/Users/misahikawa/Documents/clean_4YKN.pdb"


ligands = {
"Avapritinib": "CN1C=C(C=N1)C1=CN2N=CN=C(N3CCN(CC3)C3=NC=C(C=N3)[C@@](C)(N)C3=CC=C(F)C=C3)C2=C1",
}


box_center = [-9.63, 1.70, 20.64]
box_size = [20, 20, 20]


results_folder = Path("/Users/misahikawa/Documents/Results")
docks_folder = results_folder / "Docks"
distances_folder = results_folder / "Distances"


results_folder.mkdir(exist_ok=True)
docks_folder.mkdir(exist_ok=True)
distances_folder.mkdir(exist_ok=True)




# ========= HELPERS =========
def next_available_file(path: Path) -> Path:
   if not path.exists():
       return path
   i = 1
   while True:
       candidate = path.with_stem(f"{path.stem}{i}")
       if not candidate.exists():
           return candidate
       i += 1




def has_unsupported_elements(mol, allowed={"C","H","N","O","S","P","F","Cl","Br","I"}):
   for atom in mol.GetAtoms():
       if atom.GetSymbol() not in allowed:
           return True
   return False




# ========= CORE FUNCTIONS =========
def smiles_to_pdbqt(name, smiles, outdir):
   outdir.mkdir(exist_ok=True)


   mol = Chem.MolFromSmiles(smiles)
   if mol is None:
       print(f"WARNING: Could not parse SMILES for {name}, skipping.")
       return None


   if has_unsupported_elements(mol):
       print(f"WARNING: {name} contains unsupported elements for AutoDock Vina, skipping.")
       return None


   mol = Chem.AddHs(mol)


   try:
       status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
       if status != 0:
           raise ValueError("Embedding failed")


       AllChem.MMFFOptimizeMolecule(mol)
   except Exception as e:
       print(f"WARNING: 3D generation failed for {name}: {e}")
       return None


   tmp_pdb = outdir / f"{name}.pdb"
   Chem.MolToPDBFile(mol, str(tmp_pdb))


   pdbqt_file = outdir / f"{name}.pdbqt"
   subprocess.run(["obabel", str(tmp_pdb), "-O", str(pdbqt_file)], check=True)


   tmp_pdb.unlink()
   return pdbqt_file




def run_vina(ligand_pdbqt, outdir):
   v = Vina(sf_name="vina")
   v.set_receptor(str(protein_pdbqt))
   v.set_ligand_from_file(str(ligand_pdbqt))


   v.compute_vina_maps(center=box_center, box_size=box_size)
   v.dock(exhaustiveness=8, n_poses=5)


   out_pdbqt = outdir / f"{ligand_pdbqt.stem}_poses.pdbqt"
   out_pdb = outdir / f"{ligand_pdbqt.stem}_poses.pdb"


   v.write_poses(str(out_pdbqt), overwrite=True)
   subprocess.run(["obabel", str(out_pdbqt), "-O", str(out_pdb)], check=True)


   complex_pdb = outdir / f"{ligand_pdbqt.stem}_complex.pdb"
   merge_script = outdir / "merge_tmp.py"
   merge_script.write_text(f"""
from pymol import cmd
cmd.load("{protein_pdb}", "protein")
cmd.load("{out_pdb}", "ligand")
cmd.save("{complex_pdb}", "protein or ligand")
cmd.delete("all")
""")
   subprocess.run(["pymol", "-cq", str(merge_script)], check=True)
   merge_script.unlink()


   energies = []
   with open(out_pdbqt) as f:
       for line in f:
           if line.startswith("REMARK VINA RESULT:"):
               energies.append(float(line.split()[3]))


   ligand_pdbqt.unlink()
   return energies, out_pdb, complex_pdb




def run_pymol_distances(ligand_pdb, ligand_name, outdir, protein_pdb, top_n=50):
   outdir.mkdir(exist_ok=True)


   base_dist_file = outdir / f"{ligand_name}_distances.txt"
   dist_file = next_available_file(base_dist_file)
   script_file = outdir / f"{ligand_name}_distances.py"


   script = f"""
from pymol import cmd
cmd.load("{protein_pdb}", "protein")
cmd.load("{ligand_pdb}", "{ligand_name}")


with open("{dist_file}", "w") as f:
   f.write("LigandAtom\\tProteinAtom\\tDistance(Å)\\n")
   drug_atoms = cmd.get_model("{ligand_name}").atom
   protein_atoms = cmd.get_model("protein").atom
   distances = []


   for a1 in drug_atoms:
       for a2 in protein_atoms:
           sel1 = f"{ligand_name} and chain {{a1.chain}} and resi {{a1.resi}} and name {{a1.name}}"
           sel2 = f"protein and chain {{a2.chain}} and resi {{a2.resi}} and name {{a2.name}}"
           d = cmd.get_distance(sel1, sel2)
           distances.append((a1, a2, d))


   distances.sort(key=lambda x: x[2])


   for a1, a2, d in distances[:{top_n}]:
       f.write(
           f"{{a1.name}}_{{a1.resn}}{{a1.resi}}_{{a1.chain}}\\t"
           f"{{a2.name}}_{{a2.resn}}{{a2.resi}}_{{a2.chain}}\\t"
           f"{{d:.2f}}\\n"
       )
"""
   script_file.write_text(script)
   subprocess.run(["pymol", "-cq", str(script_file)], check=True)
   script_file.unlink()




def main():
   binding_results = next_available_file(results_folder / "binding_scores.txt")


   with open(binding_results, "w") as out:
       out.write("Ligand\tBindingEnergies(kcal/mol)\n")


       for name, smiles in ligands.items():
           print(f"=== Processing {name} ===")


           ligand_folder = docks_folder / name
           ligand_folder.mkdir(exist_ok=True)


           lig_pdbqt = smiles_to_pdbqt(name, smiles, ligand_folder)
           if lig_pdbqt is None:
               continue


           try:
               energies, docked_pdb, complex_pdb = run_vina(lig_pdbqt, ligand_folder)
           except Exception as e:
               print(f"WARNING: Docking failed for {name}: {e}")
               continue


           out.write(f"{name}\t{','.join(map(str, energies))}\n")


           run_pymol_distances(
               ligand_pdb=docked_pdb,
               ligand_name=name,
               outdir=distances_folder,
               protein_pdb=protein_pdb
           )


           print(f"Saved protein-ligand complex: {complex_pdb}")


   print("\nPipeline complete.")
   print(f"Results folder: {results_folder}")




if __name__ == "__main__":
   main()



