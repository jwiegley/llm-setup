{
  description = "hf.el — LLM model management for Emacs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        emacsWithPkgs =
          (pkgs.emacsPackagesFor pkgs.emacs-nox).emacsWithPackages
            (epkgs: [
              epkgs.package-lint
              epkgs.elisp-autofmt
              epkgs.relint
            ]);

        src = builtins.path {
          path = ./.;
          name = "hf-el-src";
          filter =
            path: type:
            let
              baseName = builtins.baseNameOf path;
            in
            baseName != ".git"
            && baseName != ".direnv"
            && baseName != "result"
            && !(pkgs.lib.hasSuffix ".elc" baseName);
        };

        mkCheck =
          name: script:
          pkgs.stdenv.mkDerivation {
            inherit name src;
            nativeBuildInputs = [ emacsWithPkgs ];
            dontConfigure = true;
            buildPhase = ''
              export HOME=$(mktemp -d)
              ${script}
            '';
            installPhase = "touch $out";
          };
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "hf-el";
          version = "0.1.0";
          inherit src;
          nativeBuildInputs = [ emacsWithPkgs ];
          buildPhase = ''
            emacs --batch -L . -f batch-byte-compile hf.el
          '';
          installPhase = ''
            mkdir -p $out/share/emacs/site-lisp
            cp hf.el hf.elc $out/share/emacs/site-lisp/
          '';
        };

        checks = {
          # Byte-compile with all warnings treated as errors
          byte-compile = mkCheck "hf-byte-compile" ''
            emacs --batch -L . \
              --eval '(setq byte-compile-error-on-warn t)' \
              -f batch-byte-compile hf.el
          '';

          # Package header and dependency lint
          package-lint = mkCheck "hf-package-lint" ''
            emacs --batch -L . \
              -l package-lint \
              -f package-lint-batch-and-exit hf.el
          '';

          # Docstring convention check (advisory — does not fail the build)
          checkdoc = mkCheck "hf-checkdoc" ''
            emacs --batch -L . \
              -l ${./scripts/run-checkdoc.el} hf.el || true
          '';

          # Regexp lint
          relint = mkCheck "hf-relint" ''
            emacs --batch -L . \
              -l relint \
              -f relint-batch hf.el
          '';

          # NOTE: elisp-autofmt format check cannot run in the Nix sandbox
          # (subprocess pipe issues). Formatting is enforced by lefthook pre-commit.
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            emacsWithPkgs
            pkgs.lefthook
          ];
          shellHook = ''
            lefthook install
          '';
        };
      }
    );
}
