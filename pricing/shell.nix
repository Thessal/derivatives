{ pkgs ? import <nixpkgs> {} }:
let
  antigravity = pkgs.callPackage ../antigravity.nix {};

in
pkgs.mkShell {
  buildInputs = [
    #pkgs.lean4
    pkgs.elan
    antigravity
    pkgs.vscode-extensions.leanprover.lean4
  ];

  shellHook = ''
    echo "Lean"
  '';
}
