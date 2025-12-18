{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixgl.url = "github:guibou/nixGL";
  };

  outputs =
    { nixpkgs, nixgl, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
            overlays = [ nixgl.overlay ];
          };
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
            ];

            env = lib.optionalAttrs pkgs.stdenv.isLinux {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
            };

            shellHook = ''
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
            '';
          };

          hpc = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
              pkgs.nixgl.auto.nixGLDefault
            ];

            LD_LIBRARY_PATH = lib.makeLibraryPath (
              pkgs.pythonManylinuxPackages.manylinux1
              ++ [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ]
            );

            shellHook = ''
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
              export LD_PRELOAD=/lib64/libcuda.so.1
            '';
          };
        }
      );
    };
}
