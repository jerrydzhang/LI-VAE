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
              pkgs.patchelf
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
              echo "🔍 Checking Ray binaries for compatibility..."

              # Get the path to the correct Nix dynamic linker
              NIX_LOADER="$(cat $NIX_CC/nix-support/dynamic-linker)"

              # Find Ray binaries (gcs_server, raylet, etc.) inside the venv
              find .venv -type f \( -name "gcs_server" -o -name "raylet" -o -name "plasma_store_server" \) 2>/dev/null | while read bin; do
                
                # Check the current interpreter of the binary
                current_interp=$(patchelf --print-interpreter "$bin" 2>/dev/null)
                
                # If it doesn't match the Nix loader, patch it
                if [ "$current_interp" != "$NIX_LOADER" ]; then
                   echo "🛠️  Patching $bin..."
                   patchelf --set-interpreter "$NIX_LOADER" "$bin"
                fi
              done
            '';
          };
        }
      );
    };
}
