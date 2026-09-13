// Dump files out of the Brood War MPQs using OpenSnowstorm's own reader
// (handles the old-format encryption/compression mpyq can't).
// Run from the directory holding StarDat.mpq/BrooDat.mpq/Patch_rt.mpq:
//   dump_bw_file <out_dir> <archive\path\file> ...
#include "data_loading.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: dump_bw_file <out_dir> <file-in-mpq>...\n");
        return 1;
    }
    auto loader = bwgame::data_loading::data_files_directory(".");
    for (int i = 2; i < argc; i++) {
        bwgame::a_vector<uint8_t> data;
        try {
            loader(data, argv[i]);
        } catch (const std::exception& e) {
            fprintf(stderr, "%s: %s\n", argv[i], e.what());
            return 2;
        }
        std::string name = argv[i];
        for (auto& c : name) if (c == '\\' || c == '/') c = '_';
        std::string out = std::string(argv[1]) + "/" + name;
        FILE* f = fopen(out.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 3; }
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
        printf("%s -> %s (%zu bytes)\n", argv[i], out.c_str(), data.size());
    }
    return 0;
}
