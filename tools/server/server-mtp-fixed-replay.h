#pragma once

#include "common.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

class server_mtp_fixed_replay {
    using json = nlohmann::json;
    struct event {
        char kind;
        uint64_t step;
        llama_pos pos;
        int meta;
        llama_tokens ids;
    };

    int request = -1;
    bool replay = false;
    std::filesystem::path tape_path;
    uint64_t prompt_hash = 0;
    int max_tokens = 0;
    int vocab_size = 0;
    std::vector<event> events;
    size_t cursor = 0;
    size_t forced_samples = 0, forced_drafts = 0, verifies3 = 0, optimized3 = 0;
    uint64_t hash = 14695981039346656037ULL;
    json expected_final;

    static void add_hash(uint64_t & h, uint64_t value) {
        h = (h ^ value) * 1099511628211ULL;
    }

    static uint64_t hash_tokens(const llama_tokens & tokens) {
        uint64_t result = 14695981039346656037ULL;
        for (llama_token token : tokens) {
            add_hash(result, uint32_t(token));
        }
        return result;
    }

    [[noreturn]] void mismatch(const char * reason) const {
        throw std::runtime_error("MTP fixed replay mismatch at event " + std::to_string(cursor) + ": " + reason);
    }

    const event & exchange(char kind, uint64_t step, llama_pos pos, int meta, const llama_tokens & ids) {
        if (!replay) {
            events.push_back({kind, step, pos, meta, ids});
        } else if (cursor >= events.size() || events[cursor].kind != kind || events[cursor].step != step ||
                events[cursor].pos != pos || events[cursor].meta != meta) {
            mismatch("phase, generation index, sequence position, or batch shape changed");
        }
        const auto & row = events[cursor++];
        add_hash(hash, kind);
        add_hash(hash, step);
        add_hash(hash, uint32_t(pos));
        add_hash(hash, uint32_t(meta));
        add_hash(hash, row.ids.size());
        for (llama_token id : row.ids) {
            add_hash(hash, uint32_t(id));
        }
        return row;
    }

public:
    static const char * mode_path() {
        static const char * path = std::getenv("LLAMA_MTP_FIXED_REPLAY_MODE_FILE");
        return path && *path ? path : nullptr;
    }

    bool active() const { return request >= 0; }
    bool active(int id) const { return active() && request == id; }
    bool replaying() const { return replay; }

    void begin(int id, bool supported, const llama_tokens & prompt, int predict, int n_vocab) {
        *this = {};
        const auto path = std::filesystem::u8path(mode_path());
        std::ifstream mode_file(path);
        if (!mode_file && !std::filesystem::exists(path)) {
            return;
        }
        std::string line;
        if (!std::getline(mode_file, line)) {
            throw std::runtime_error("MTP fixed replay: could not read mode file");
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line == "off") {
            return;
        }
        replay = line.compare(0, 7, "replay ") == 0;
        if (!replay && line.compare(0, 7, "record ") != 0) {
            throw std::runtime_error("MTP fixed replay: expected off, record PATH, or replay PATH");
        }
        if (!supported || predict < 1 || predict > 4096 || n_vocab < 1) {
            throw std::runtime_error("MTP fixed replay requires one slot, fixed MTP3 rollback, greedy text completion, return_tokens, and 1-4096 output tokens");
        }
        tape_path = std::filesystem::u8path(line.substr(7));
        if (!tape_path.is_absolute()) {
            throw std::runtime_error("MTP fixed replay: tape path must be absolute");
        }
        prompt_hash = hash_tokens(prompt);
        max_tokens = predict;
        vocab_size = n_vocab;
        if (replay) {
            if (std::filesystem::file_size(tape_path) > 8 * 1024 * 1024) {
                mismatch("tape exceeds the size limit");
            }
            std::ifstream tape(tape_path);
            const auto doc = json::parse(tape);
            if (!doc.at("version").is_number_integer() || doc.at("version") != 1 ||
                    !doc.at("prompt_hash").is_number_unsigned() || doc.at("prompt_hash") != prompt_hash ||
                    !doc.at("n_predict").is_number_integer() || doc.at("n_predict") != predict ||
                    !doc.at("vocab_size").is_number_integer() || doc.at("vocab_size") != vocab_size) {
                mismatch("request settings or formatted prompt tokens changed");
            }
            const auto & rows = doc.at("events");
            if (!rows.is_array() || rows.size() > size_t(predict) * 4 + 8) {
                mismatch("invalid event count");
            }
            for (const auto & row : rows) {
                if (!row.is_array() || row.size() != 5 || !row[0].is_string() || !row[1].is_number_unsigned() ||
                        row[1] > predict || !row[2].is_number_integer() || row[2] < 0 || row[2] > INT32_MAX ||
                        !row[3].is_number_integer() || row[3] < 0 || row[3] > INT32_MAX ||
                        !row[4].is_array() || row[4].size() > 3) {
                    mismatch("invalid event in tape");
                }
                const auto kind = row[0].get<std::string>();
                if (kind.size() != 1 || std::string_view("DBSV").find(kind.front()) == std::string_view::npos) {
                    mismatch("invalid event kind");
                }
                llama_tokens ids;
                for (const auto & id : row[4]) {
                    if (!id.is_number_integer() || id < 0 || id >= vocab_size) {
                        mismatch("invalid token ID in tape");
                    }
                    ids.push_back(id.get<llama_token>());
                }
                events.push_back({kind.front(), row[1].get<uint64_t>(), row[2].get<llama_pos>(), row[3].get<int>(), std::move(ids)});
            }
            expected_final = doc.at("final");
        }
        request = id;
    }

    void drafts(uint64_t step, llama_pos pos, llama_token anchor, llama_tokens & ids) {
        const auto & row = exchange('D', step, pos, anchor, ids);
        if (row.ids.size() != ids.size()) {
            mismatch("natural drafter proposed a different width");
        }
        if (replay) {
            for (size_t i = 0; i < ids.size(); ++i) {
                forced_drafts += ids[i] != row.ids[i];
                ids[i] = row.ids[i];
            }
        }
    }

    void verify_batch(uint64_t step, llama_pos pos, bool optimized, const llama_tokens & ids) {
        const auto & row = exchange('B', step, pos, optimized, ids);
        if (row.ids != ids) {
            mismatch("verify input IDs or width changed");
        }
        verifies3 += ids.size() == 3;
        optimized3 += ids.size() == 3 && optimized;
    }

    llama_token select(llama_token natural, const llama_tokens & chosen, size_t index) {
        if (index >= chosen.size()) {
            mismatch("verify acceptance exceeded the recorded prefix");
        }
        forced_samples += natural != chosen[index];
        return chosen[index];
    }

    llama_token single(uint64_t step, llama_pos pos, llama_token natural) {
        const auto & row = exchange('S', step, pos, 1, {natural});
        return select(natural, row.ids, 0);
    }

    const llama_tokens & verify(uint64_t step, llama_pos pos, size_t width, const llama_tokens & accepted) {
        return exchange('V', step, pos, int(width), accepted).ids;
    }

    void finish(uint64_t generated, uint64_t proposed, uint64_t accepted, int stop, const llama_tokens & emitted) {
        const json final = {{"generated", generated}, {"proposed", proposed}, {"accepted", accepted}, {"stop", stop},
                {"emitted", emitted}, {"event_hash", hash}, {"verify3", verifies3}, {"optimized3", optimized3}};
        if (replay) {
            if (cursor != events.size() || final != expected_final) {
                mismatch("request ended with different totals, stop reason, or emitted IDs");
            }
        } else {
            json rows = json::array();
            for (const auto & row : events) {
                rows.push_back({std::string(1, row.kind), row.step, row.pos, row.meta, row.ids});
            }
            const json doc = {{"version", 1}, {"prompt_hash", prompt_hash}, {"n_predict", max_tokens},
                    {"vocab_size", vocab_size}, {"events", std::move(rows)}, {"final", final}};
            auto temporary = tape_path;
            temporary += ".tmp";
            std::ofstream tape(temporary, std::ios::trunc);
            tape << doc.dump() << '\n';
            tape.close();
            if (!tape) {
                throw std::runtime_error("MTP fixed replay: could not write tape");
            }
            std::filesystem::rename(temporary, tape_path);
        }
        LOG_INF("mtp_fixed_replay: mode=%s request=%d events=%zu hash=%llu generated=%llu proposed=%llu accepted=%llu"
                " verify3=%zu optimized3=%zu forced_samples=%zu forced_drafts=%zu\n",
                replay ? "replay" : "record", request, cursor, (unsigned long long) hash, (unsigned long long) generated,
                (unsigned long long) proposed, (unsigned long long) accepted, verifies3, optimized3, forced_samples, forced_drafts);
        request = -1;
    }
};
