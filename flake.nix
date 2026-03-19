{
  description = "llm-setup.el — LLM model management for Emacs";

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
          name = "llm-setup-el-src";
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
          pname = "llm-setup-el";
          version = "0.1.0";
          inherit src;
          nativeBuildInputs = [ emacsWithPkgs ];
          buildPhase = ''
            emacs --batch -L . -f batch-byte-compile llm-setup.el
          '';
          installPhase = ''
            mkdir -p $out/share/emacs/site-lisp
            cp llm-setup.el llm-setup.elc $out/share/emacs/site-lisp/
          '';
        };

        checks = {
          # Byte-compile with all warnings treated as errors
          byte-compile = mkCheck "llm-setup-byte-compile" ''
            emacs --batch -L . \
              --eval '(setq byte-compile-error-on-warn t)' \
              -f batch-byte-compile llm-setup.el
          '';

          # Package header and dependency lint
          package-lint = mkCheck "llm-setup-package-lint" ''
            emacs --batch -L . \
              -l package-lint \
              -f package-lint-batch-and-exit llm-setup.el
          '';

          # Docstring convention check (advisory — does not fail the build)
          checkdoc = mkCheck "llm-setup-checkdoc" ''
            emacs --batch -L . \
              -l ${./scripts/run-checkdoc.el} llm-setup.el || true
          '';

          # Regexp lint
          relint = mkCheck "llm-setup-relint" ''
            emacs --batch -L . \
              -l relint \
              -f relint-batch llm-setup.el
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
