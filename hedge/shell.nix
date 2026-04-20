let
  pkgs = import <nixpkgs> {
    overlays = [
      (import (fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz"))
    ];
  };
  rustVersion = "1.91.1";
  myRust = pkgs.rust-bin.stable.${rustVersion}.default.override {
    extensions = ["rust-src" "rust-analyzer"];
  };

  antigravity = pkgs.callPackage ../antigravity.nix {};
  gpuEnvVar = pkgs.callPackage ../gpu.nix {};

in pkgs.mkShell{
  packages = [
    #pkgs.antigravity
    pkgs.google-chrome
    antigravity
  ] ++ (with pkgs; [
    python313 ( with python313.pkgs; [requests pandas linearmodels numpy scipy statsmodels sympy lxml yfinance boto3 fsspec s3fs zstandard] )
    jq
    python313 ( with python313.pkgs; [ipykernel matplotlib torch] )
  ]) ++ 
    [ myRust ] ++ (with pkgs; [
    cargo rustc gcc rustfmt clippy rust-analyzer gdb
  ]) ; 
  shellHook = gpuEnvVar.rocm + ''
    LANG=C
  '';
}
