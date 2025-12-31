const std = @import("std");

const Allocator = std.mem.Allocator;

const Task = enum { classification, regression };

const Method = enum { pearson, spearman, mi };

const MethodEval = struct {
    method: Method,
    k: usize,
    cv_score_mean: f64,
    cv_score_std: f64,
    stability_jaccard: f64,
    runtime_sec: f64,
};

const MetaFeatures = struct {
    n_samples: usize,
    n_features: usize,
    p_over_n: f64,
    missing_rate: f64,
    y_unique: usize,
    approx_task: Task,
};

fn methodName(m: Method) []const u8 {
    return switch (m) {
        .pearson => "pearson_abs",
        .spearman => "spearman_abs",
        .mi => "mutual_info(binned)",
    };
}

fn parseArgs(alloc: Allocator) !Args {
    var it = std.process.args();
    _ = it.next(); // exe

    var args = Args{
        .csv_path = null,
        .target = null,
        .k = 20,
        .cv = 5,
        .store = "experience.jsonl",
        .methods = .{ .pearson = true, .spearman = true, .mi = true },
    };

    while (it.next()) |a| {
        if (std.mem.eql(u8, a, "--csv")) {
            args.csv_path = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, a, "--target")) {
            args.target = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, a, "--k")) {
            const v = it.next() orelse return error.InvalidArgs;
            args.k = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--cv")) {
            const v = it.next() orelse return error.InvalidArgs;
            args.cv = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--store")) {
            args.store = it.next() orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, a, "--methods")) {
            const v = it.next() orelse return error.InvalidArgs;
            args.methods = try parseMethods(v);
        } else if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            return error.Help;
        } else {
            // ignore unknown for forward compatibility
        }
    }

    if (args.csv_path == null or args.target == null) return error.InvalidArgs;
    _ = alloc;
    return args;
}

const MethodsMask = struct { pearson: bool, spearman: bool, mi: bool };

const Args = struct {
    csv_path: ?[]const u8,
    target: ?[]const u8,
    k: usize,
    cv: usize,
    store: []const u8,
    methods: MethodsMask,
};

fn parseMethods(s: []const u8) !MethodsMask {
    var out = MethodsMask{ .pearson = false, .spearman = false, .mi = false };
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |tok_raw| {
        const tok = std.mem.trim(u8, tok_raw, " \t\r\n");
        if (tok.len == 0) continue;
        if (std.ascii.eqlIgnoreCase(tok, "pearson")) out.pearson = true else if (std.ascii.eqlIgnoreCase(tok, "spearman")) out.spearman = true else if (std.ascii.eqlIgnoreCase(tok, "mi")) out.mi = true else return error.InvalidArgs;
    }
    return out;
}

fn printUsage() void {
    std.debug.print(
        \\mtbmt_benchmark (zig_version)
        \\
        \\Usage:
        \\  mtbmt_benchmark --csv <path> --target <col> [--k 20] [--cv 5] [--store <jsonl>] [--methods pearson,spearman,mi]
        \\
        \\Notes:
        \\  - CSV must contain a header row.
        \\  - All feature columns are parsed as f64; non-numeric becomes NaN.
        \\  - MI is binned approximation.
        \\
        \\
    , .{});
}

const Dataset = struct {
    X: []f64, // row-major: n * p
    y: []f64, // n
    feature_names: [][]const u8,
    n: usize,
    p: usize,
    y_unique: usize,
    y_is_int: bool,
    missing_rate: f64,
    task: Task,
};

fn isIntLike(v: f64) bool {
    if (!std.math.isFinite(v)) return false;
    const r = @round(v);
    return std.math.fabs(v - r) <= 1e-9;
}

fn inferTask(y: []const f64) TaskInfo {
    var map = std.AutoHashMap(i64, void).init(std.heap.page_allocator);
    defer map.deinit();
    var y_is_int = true;
    for (y) |v| {
        if (!isIntLike(v)) {
            y_is_int = false;
            break;
        }
        _ = map.put(@as(i64, @intFromFloat(@round(v))), {}) catch {};
        if (map.count() > 20 and map.count() > (y.len / 10)) break;
    }
    const y_unique = map.count();
    const task: Task = if (y_is_int and y_unique <= @max(@as(usize, 20), y.len / 10)) .classification else .regression;
    return .{ .task = task, .y_unique = y_unique, .y_is_int = y_is_int };
}

const TaskInfo = struct { task: Task, y_unique: usize, y_is_int: bool };

fn loadCsv(alloc: Allocator, path: []const u8, target_col: []const u8) !Dataset {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(alloc, 1 << 30);
    defer alloc.free(content);

    var lines_it = std.mem.splitScalar(u8, content, '\n');
    const header_line_raw = lines_it.next() orelse return error.InvalidCsv;
    const header_line = std.mem.trimRight(u8, header_line_raw, "\r");
    var headers = std.ArrayList([]const u8).init(alloc);
    defer headers.deinit();
    try splitCsvLine(alloc, header_line, &headers);

    var target_idx: ?usize = null;
    for (headers.items, 0..) |h, i| {
        if (std.mem.eql(u8, h, target_col)) {
            target_idx = i;
            break;
        }
    }
    if (target_idx == null) return error.TargetNotFound;
    const t_idx = target_idx.?;

    // Feature names (exclude target)
    var feature_names = std.ArrayList([]const u8).init(alloc);
    errdefer {
        for (feature_names.items) |s| alloc.free(s);
        feature_names.deinit();
    }
    for (headers.items, 0..) |h, i| {
        if (i == t_idx) continue;
        try feature_names.append(try alloc.dupe(u8, h));
    }

    // Parse rows
    var X = std.ArrayList(f64).init(alloc);
    var y = std.ArrayList(f64).init(alloc);
    errdefer {
        X.deinit();
        y.deinit();
    }

    var missing_count: usize = 0;
    var total_cells: usize = 0;
    while (lines_it.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        if (std.mem.trim(u8, line, " \t").len == 0) continue;

        var cols = std.ArrayList([]const u8).init(alloc);
        defer cols.deinit();
        try splitCsvLine(alloc, line, &cols);

        if (cols.items.len != headers.items.len) continue; // skip malformed

        var yi: f64 = parseF64OrNan(cols.items[t_idx]);
        try y.append(yi);

        for (cols.items, 0..) |c, i| {
            if (i == t_idx) continue;
            const v = parseF64OrNan(c);
            total_cells += 1;
            if (std.math.isNan(v)) missing_count += 1;
            try X.append(v);
        }
    }

    const n = y.items.len;
    const p = feature_names.items.len;
    if (n == 0 or p == 0) return error.InvalidCsv;
    if (X.items.len != n * p) return error.InvalidCsv;

    const ti = inferTask(y.items);
    const missing_rate = @as(f64, @floatFromInt(missing_count)) / @as(f64, @floatFromInt(@max(total_cells, 1)));

    return Dataset{
        .X = try X.toOwnedSlice(),
        .y = try y.toOwnedSlice(),
        .feature_names = try feature_names.toOwnedSlice(),
        .n = n,
        .p = p,
        .y_unique = ti.y_unique,
        .y_is_int = ti.y_is_int,
        .missing_rate = missing_rate,
        .task = ti.task,
    };
}

fn splitCsvLine(alloc: Allocator, line: []const u8, out: *std.ArrayList([]const u8)) !void {
    // very small CSV splitter (no escaped quotes)
    _ = alloc;
    var start: usize = 0;
    var in_quotes = false;
    var i: usize = 0;
    while (i <= line.len) : (i += 1) {
        const end_reached = i == line.len;
        const ch: u8 = if (!end_reached) line[i] else 0;
        if (!end_reached and ch == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (end_reached or (!in_quotes and ch == ',')) {
            const raw = line[start..i];
            const trimmed = std.mem.trim(u8, raw, " \t");
            // remove surrounding quotes if present
            const cell = if (trimmed.len >= 2 and trimmed[0] == '"' and trimmed[trimmed.len - 1] == '"')
                trimmed[1 .. trimmed.len - 1]
            else
                trimmed;
            try out.append(cell);
            start = i + 1;
        }
    }
}

fn parseF64OrNan(s: []const u8) f64 {
    const t = std.mem.trim(u8, s, " \t");
    if (t.len == 0) return std.math.nan(f64);
    return std.fmt.parseFloat(f64, t) catch std.math.nan(f64);
}

fn pearsonAbsScores(alloc: Allocator, d: Dataset) ![]f64 {
    var scores = try alloc.alloc(f64, d.p);
    const n_f = @as(f64, @floatFromInt(d.n));

    // y mean/std
    var y_sum: f64 = 0;
    var y_cnt: usize = 0;
    for (d.y) |v| {
        if (std.math.isNan(v)) continue;
        y_sum += v;
        y_cnt += 1;
    }
    const y_mean = y_sum / @as(f64, @floatFromInt(@max(y_cnt, 1)));
    var y_var: f64 = 0;
    for (d.y) |v| {
        if (std.math.isNan(v)) continue;
        y_var += (v - y_mean) * (v - y_mean);
    }
    const y_std = std.math.sqrt(y_var / @as(f64, @floatFromInt(@max(y_cnt, 1)))) + 1e-12;

    var j: usize = 0;
    while (j < d.p) : (j += 1) {
        var x_sum: f64 = 0;
        var x_cnt: usize = 0;
        var i: usize = 0;
        while (i < d.n) : (i += 1) {
            const x = d.X[i * d.p + j];
            if (std.math.isNan(x)) continue;
            x_sum += x;
            x_cnt += 1;
        }
        const x_mean = x_sum / @as(f64, @floatFromInt(@max(x_cnt, 1)));

        var x_var: f64 = 0;
        var cov: f64 = 0;
        i = 0;
        while (i < d.n) : (i += 1) {
            const x = d.X[i * d.p + j];
            const yv = d.y[i];
            if (std.math.isNan(x) or std.math.isNan(yv)) continue;
            const dx = x - x_mean;
            const dy = yv - y_mean;
            x_var += dx * dx;
            cov += dx * dy;
        }
        const x_std = std.math.sqrt(x_var / @as(f64, @floatFromInt(@max(x_cnt, 1)))) + 1e-12;
        const corr = (cov / n_f) / (x_std * y_std);
        scores[j] = std.math.fabs(corr);
    }

    return scores;
}

fn rankData(alloc: Allocator, v: []const f64) ![]f64 {
    const n = v.len;
    var idx = try alloc.alloc(usize, n);
    defer alloc.free(idx);
    var i: usize = 0;
    while (i < n) : (i += 1) idx[i] = i;

    std.sort.pdq(usize, idx, v, struct {
        fn lessThan(ctx: []const f64, a: usize, b: usize) bool {
            return ctx[a] < ctx[b];
        }
    }.lessThan);

    var ranks = try alloc.alloc(f64, n);
    i = 0;
    while (i < n) : (i += 1) ranks[i] = @as(f64, @floatFromInt(i + 1));

    // average ties (simple)
    var k: usize = 0;
    while (k < n) {
        var j = k;
        while (j + 1 < n and v[idx[j + 1]] == v[idx[k]]) : (j += 1) {}
        if (j > k) {
            const avg = (@as(f64, @floatFromInt(k + 1)) + @as(f64, @floatFromInt(j + 1))) / 2.0;
            var t = k;
            while (t <= j) : (t += 1) ranks[idx[t]] = avg;
        }
        k = j + 1;
    }
    return ranks;
}

fn spearmanAbsScores(alloc: Allocator, d: Dataset) ![]f64 {
    var y_rank = try rankData(alloc, d.y);
    defer alloc.free(y_rank);

    var scores = try alloc.alloc(f64, d.p);
    var col = try alloc.alloc(f64, d.n);
    defer alloc.free(col);

    var j: usize = 0;
    while (j < d.p) : (j += 1) {
        var i: usize = 0;
        while (i < d.n) : (i += 1) col[i] = d.X[i * d.p + j];
        const x_rank = try rankData(alloc, col);
        defer alloc.free(x_rank);

        // Pearson on ranks
        var xr_sum: f64 = 0;
        var yr_sum: f64 = 0;
        i = 0;
        while (i < d.n) : (i += 1) {
            xr_sum += x_rank[i];
            yr_sum += y_rank[i];
        }
        const xr_mean = xr_sum / @as(f64, @floatFromInt(d.n));
        const yr_mean = yr_sum / @as(f64, @floatFromInt(d.n));

        var cov: f64 = 0;
        var xvar: f64 = 0;
        var yvar: f64 = 0;
        i = 0;
        while (i < d.n) : (i += 1) {
            const dx = x_rank[i] - xr_mean;
            const dy = y_rank[i] - yr_mean;
            cov += dx * dy;
            xvar += dx * dx;
            yvar += dy * dy;
        }
        const denom = (std.math.sqrt(xvar) + 1e-12) * (std.math.sqrt(yvar) + 1e-12);
        scores[j] = std.math.fabs(cov / denom);
    }
    return scores;
}

fn miBinnedScores(alloc: Allocator, d: Dataset, bins: usize) ![]f64 {
    var scores = try alloc.alloc(f64, d.p);

    // y bins
    var y_min = std.math.inf(f64);
    var y_max = -std.math.inf(f64);
    for (d.y) |v| {
        if (std.math.isNan(v)) continue;
        y_min = @min(y_min, v);
        y_max = @max(y_max, v);
    }
    const y_range = @max(y_max - y_min, 1e-12);

    var yb = try alloc.alloc(usize, d.n);
    defer alloc.free(yb);
    for (d.y, 0..) |v, i| {
        if (std.math.isNan(v)) {
            yb[i] = 0;
        } else {
            const u = (v - y_min) / y_range;
            const bi = @min(@as(usize, @intFromFloat(@floor(u * @as(f64, @floatFromInt(bins))))), bins - 1);
            yb[i] = bi;
        }
    }

    var j: usize = 0;
    while (j < d.p) : (j += 1) {
        // x min/max
        var x_min = std.math.inf(f64);
        var x_max = -std.math.inf(f64);
        var i: usize = 0;
        while (i < d.n) : (i += 1) {
            const x = d.X[i * d.p + j];
            if (std.math.isNan(x)) continue;
            x_min = @min(x_min, x);
            x_max = @max(x_max, x);
        }
        const x_range = @max(x_max - x_min, 1e-12);

        // counts
        var joint = try alloc.alloc(usize, bins * bins);
        defer alloc.free(joint);
        @memset(joint, 0);
        var x_cnt = try alloc.alloc(usize, bins);
        defer alloc.free(x_cnt);
        @memset(x_cnt, 0);
        var y_cnt = try alloc.alloc(usize, bins);
        defer alloc.free(y_cnt);
        @memset(y_cnt, 0);

        i = 0;
        while (i < d.n) : (i += 1) {
            const x = d.X[i * d.p + j];
            const by = yb[i];
            const bx: usize = if (std.math.isNan(x)) 0 else blk: {
                const u = (x - x_min) / x_range;
                break :blk @min(@as(usize, @intFromFloat(@floor(u * @as(f64, @floatFromInt(bins))))), bins - 1);
            };
            joint[bx * bins + by] += 1;
            x_cnt[bx] += 1;
            y_cnt[by] += 1;
        }

        const n_f = @as(f64, @floatFromInt(d.n));
        var mi: f64 = 0;
        var bx: usize = 0;
        while (bx < bins) : (bx += 1) {
            var by: usize = 0;
            while (by < bins) : (by += 1) {
                const c = joint[bx * bins + by];
                if (c == 0) continue;
                const pxy = @as(f64, @floatFromInt(c)) / n_f;
                const px = @as(f64, @floatFromInt(x_cnt[bx])) / n_f;
                const py = @as(f64, @floatFromInt(y_cnt[by])) / n_f;
                mi += pxy * std.math.log(pxy / (px * py + 1e-18) + 1e-18);
            }
        }
        scores[j] = mi;
    }
    return scores;
}

fn topKIndices(alloc: Allocator, scores: []const f64, k: usize) ![]usize {
    const p = scores.len;
    var idx = try alloc.alloc(usize, p);
    var i: usize = 0;
    while (i < p) : (i += 1) idx[i] = i;
    std.sort.pdq(usize, idx, scores, struct {
        fn lessThan(ctx: []const f64, a: usize, b: usize) bool {
            return ctx[a] > ctx[b];
        }
    }.lessThan);
    const kk = @min(k, p);
    const out = try alloc.alloc(usize, kk);
    @memcpy(out, idx[0..kk]);
    alloc.free(idx);
    return out;
}

fn jaccard(a: []const usize, b: []const usize) f64 {
    var sa = std.AutoHashMap(usize, void).init(std.heap.page_allocator);
    defer sa.deinit();
    for (a) |v| _ = sa.put(v, {}) catch {};
    var inter: usize = 0;
    var union: usize = sa.count();
    for (b) |v| {
        if (sa.contains(v)) {
            inter += 1;
        } else {
            union += 1;
        }
    }
    if (union == 0) return 1.0;
    return @as(f64, @floatFromInt(inter)) / @as(f64, @floatFromInt(union));
}

fn stratifiedFolds(alloc: Allocator, y: []const f64, kfold: usize, seed: u64) ![][]usize {
    // binary/int labels assumed
    var class0 = std.ArrayList(usize).init(alloc);
    var class1 = std.ArrayList(usize).init(alloc);
    defer class0.deinit();
    defer class1.deinit();
    for (y, 0..) |v, i| {
        if (isIntLike(v) and @as(i64, @intFromFloat(@round(v))) == 1) {
            try class1.append(i);
        } else {
            try class0.append(i);
        }
    }
    var prng = std.rand.DefaultPrng.init(seed);
    prng.random().shuffle(usize, class0.items);
    prng.random().shuffle(usize, class1.items);

    var folds = try alloc.alloc([]usize, kfold);
    for (folds, 0..) |*f, _| f.* = &[_]usize{};

    // fill each fold
    var tmp = std.ArrayList(usize).init(alloc);
    defer tmp.deinit();
    var f: usize = 0;
    while (f < kfold) : (f += 1) {
        tmp.clearRetainingCapacity();
        var i0: usize = f;
        while (i0 < class0.items.len) : (i0 += kfold) try tmp.append(class0.items[i0]);
        var i1: usize = f;
        while (i1 < class1.items.len) : (i1 += kfold) try tmp.append(class1.items[i1]);
        folds[f] = try tmp.toOwnedSlice();
    }
    return folds;
}

fn kfoldFolds(alloc: Allocator, n: usize, kfold: usize, seed: u64) ![][]usize {
    var idx = try alloc.alloc(usize, n);
    var i: usize = 0;
    while (i < n) : (i += 1) idx[i] = i;
    var prng = std.rand.DefaultPrng.init(seed);
    prng.random().shuffle(usize, idx);

    var folds = try alloc.alloc([]usize, kfold);
    var f: usize = 0;
    while (f < kfold) : (f += 1) {
        var tmp = std.ArrayList(usize).init(alloc);
        errdefer tmp.deinit();
        var j: usize = f;
        while (j < n) : (j += kfold) try tmp.append(idx[j]);
        folds[f] = try tmp.toOwnedSlice();
    }
    alloc.free(idx);
    return folds;
}

fn aucScore(alloc: Allocator, y_true: []const f64, y_score: []const f64) !f64 {
    const n = y_true.len;
    var idx = try alloc.alloc(usize, n);
    defer alloc.free(idx);
    var i: usize = 0;
    while (i < n) : (i += 1) idx[i] = i;
    std.sort.pdq(usize, idx, y_score, struct {
        fn lessThan(ctx: []const f64, a: usize, b: usize) bool {
            return ctx[a] < ctx[b];
        }
    }.lessThan);

    var rank_sum_pos: f64 = 0;
    var n_pos: f64 = 0;
    var n_neg: f64 = 0;
    i = 0;
    while (i < n) : (i += 1) {
        const yi = y_true[idx[i]];
        const r = @as(f64, @floatFromInt(i + 1));
        if (isIntLike(yi) and @as(i64, @intFromFloat(@round(yi))) == 1) {
            rank_sum_pos += r;
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }
    if (n_pos == 0 or n_neg == 0) return 0.5;
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg);
}

fn logisticTrainPredictAuc(
    alloc: Allocator,
    d: Dataset,
    train_idx: []const usize,
    test_idx: []const usize,
    feat_idx: []const usize,
) !f64 {
    const k = feat_idx.len;
    var w = try alloc.alloc(f64, k + 1); // bias + weights
    defer alloc.free(w);
    @memset(w, 0);

    const lr: f64 = 0.1;
    const l2: f64 = 1e-3;
    const iters: usize = 200;

    var grad = try alloc.alloc(f64, k + 1);
    defer alloc.free(grad);

    var iter: usize = 0;
    while (iter < iters) : (iter += 1) {
        @memset(grad, 0);
        for (train_idx) |ii| {
            const yv = d.y[ii];
            const y01: f64 = if (isIntLike(yv) and @as(i64, @intFromFloat(@round(yv))) == 1) 1.0 else 0.0;
            var z: f64 = w[0];
            var j: usize = 0;
            while (j < k) : (j += 1) {
                const x = d.X[ii * d.p + feat_idx[j]];
                z += w[j + 1] * (if (std.math.isNan(x)) 0.0 else x);
            }
            const p = 1.0 / (1.0 + std.math.exp(-z));
            const err = p - y01;
            grad[0] += err;
            j = 0;
            while (j < k) : (j += 1) {
                const x = d.X[ii * d.p + feat_idx[j]];
                grad[j + 1] += err * (if (std.math.isNan(x)) 0.0 else x);
            }
        }
        const ntr = @as(f64, @floatFromInt(@max(train_idx.len, 1)));
        var g: usize = 0;
        while (g < k + 1) : (g += 1) {
            var reg = 0.0;
            if (g > 0) reg = l2 * w[g];
            w[g] -= lr * ((grad[g] / ntr) + reg);
        }
    }

    // predict on test
    var ys = try alloc.alloc(f64, test_idx.len);
    defer alloc.free(ys);
    var yt = try alloc.alloc(f64, test_idx.len);
    defer alloc.free(yt);
    var t: usize = 0;
    while (t < test_idx.len) : (t += 1) {
        const ii = test_idx[t];
        yt[t] = d.y[ii];
        var z: f64 = w[0];
        var j: usize = 0;
        while (j < k) : (j += 1) {
            const x = d.X[ii * d.p + feat_idx[j]];
            z += w[j + 1] * (if (std.math.isNan(x)) 0.0 else x);
        }
        ys[t] = 1.0 / (1.0 + std.math.exp(-z));
    }
    return aucScore(alloc, yt, ys);
}

fn r2Score(y_true: []const f64, y_pred: []const f64) f64 {
    var mean: f64 = 0;
    for (y_true) |v| mean += v;
    mean /= @as(f64, @floatFromInt(@max(y_true.len, 1)));
    var ss_tot: f64 = 0;
    var ss_res: f64 = 0;
    for (y_true, 0..) |v, i| {
        ss_tot += (v - mean) * (v - mean);
        ss_res += (v - y_pred[i]) * (v - y_pred[i]);
    }
    return 1.0 - ss_res / (ss_tot + 1e-12);
}

fn ridgeTrainPredictR2(
    alloc: Allocator,
    d: Dataset,
    train_idx: []const usize,
    test_idx: []const usize,
    feat_idx: []const usize,
) !f64 {
    const k = feat_idx.len;
    const alpha: f64 = 1.0;

    // Build normal equations: (X^T X + alpha I) w = X^T y
    var A = try alloc.alloc(f64, k * k);
    defer alloc.free(A);
    @memset(A, 0);
    var b = try alloc.alloc(f64, k);
    defer alloc.free(b);
    @memset(b, 0);

    for (train_idx) |ii| {
        const yi = d.y[ii];
        var j: usize = 0;
        while (j < k) : (j += 1) {
            const xj = d.X[ii * d.p + feat_idx[j]];
            const vxj = if (std.math.isNan(xj)) 0.0 else xj;
            b[j] += vxj * yi;
            var t: usize = 0;
            while (t < k) : (t += 1) {
                const xt = d.X[ii * d.p + feat_idx[t]];
                const vxt = if (std.math.isNan(xt)) 0.0 else xt;
                A[j * k + t] += vxj * vxt;
            }
        }
    }
    // add alpha I
    var j: usize = 0;
    while (j < k) : (j += 1) A[j * k + j] += alpha;

    // Solve A w = b via Gauss-Jordan
    var w = try alloc.alloc(f64, k);
    defer alloc.free(w);
    try solveLinearSystemInPlace(alloc, A, b, w, k);

    // predict
    var yp = try alloc.alloc(f64, test_idx.len);
    defer alloc.free(yp);
    var yt = try alloc.alloc(f64, test_idx.len);
    defer alloc.free(yt);
    var i: usize = 0;
    while (i < test_idx.len) : (i += 1) {
        const ii = test_idx[i];
        yt[i] = d.y[ii];
        var pred: f64 = 0;
        var t: usize = 0;
        while (t < k) : (t += 1) {
            const x = d.X[ii * d.p + feat_idx[t]];
            pred += w[t] * (if (std.math.isNan(x)) 0.0 else x);
        }
        yp[i] = pred;
    }
    return r2Score(yt, yp);
}

fn solveLinearSystemInPlace(alloc: Allocator, A: []f64, b: []f64, out: []f64, n: usize) !void {
    // Augmented matrix [A|b], Gauss-Jordan
    _ = alloc;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        // pivot
        var pivot = A[i * n + i];
        if (std.math.fabs(pivot) < 1e-12) pivot = 1e-12;
        var j: usize = 0;
        while (j < n) : (j += 1) A[i * n + j] /= pivot;
        b[i] /= pivot;

        // eliminate other rows
        var r: usize = 0;
        while (r < n) : (r += 1) {
            if (r == i) continue;
            const factor = A[r * n + i];
            if (std.math.fabs(factor) < 1e-18) continue;
            j = 0;
            while (j < n) : (j += 1) A[r * n + j] -= factor * A[i * n + j];
            b[r] -= factor * b[i];
        }
    }
    @memcpy(out, b[0..n]);
}

fn evaluateMethod(alloc: Allocator, d: Dataset, method: Method, k: usize, cv: usize) !MethodEval {
    var timer = try std.time.Timer.start();
    const folds = if (d.task == .classification) try stratifiedFolds(alloc, d.y, cv, 0) else try kfoldFolds(alloc, d.n, cv, 0);
    defer {
        for (folds) |f| alloc.free(f);
        alloc.free(folds);
    }

    var fold_scores = try alloc.alloc(f64, cv);
    defer alloc.free(fold_scores);
    var fold_topk = try alloc.alloc([]usize, cv);
    defer {
        for (fold_topk) |t| alloc.free(t);
        alloc.free(fold_topk);
    }

    var f: usize = 0;
    while (f < cv) : (f += 1) {
        const test_idx = folds[f];
        // train_idx = all others
        var train = std.ArrayList(usize).init(alloc);
        defer train.deinit();
        var g: usize = 0;
        while (g < cv) : (g += 1) {
            if (g == f) continue;
            try train.appendSlice(folds[g]);
        }
        const train_idx = train.items;

        // compute scores on train subset
        var sub = try makeSubsetDataset(alloc, d, train_idx);
        defer sub.deinit(alloc);
        const scores = switch (method) {
            .pearson => try pearsonAbsScores(alloc, sub.d),
            .spearman => try spearmanAbsScores(alloc, sub.d),
            .mi => try miBinnedScores(alloc, sub.d, 16),
        };
        defer alloc.free(scores);

        const topk = try topKIndices(alloc, scores, k);
        fold_topk[f] = topk;

        // train/predict on full dataset using selected features
        const s = if (d.task == .classification)
            try logisticTrainPredictAuc(alloc, d, train_idx, test_idx, topk)
        else
            try ridgeTrainPredictR2(alloc, d, train_idx, test_idx, topk);
        fold_scores[f] = s;
    }

    // stability: average pairwise jaccard
    var jac_sum: f64 = 0;
    var cnt: usize = 0;
    var i: usize = 0;
    while (i < cv) : (i += 1) {
        var j: usize = i + 1;
        while (j < cv) : (j += 1) {
            jac_sum += jaccard(fold_topk[i], fold_topk[j]);
            cnt += 1;
        }
    }
    const stability = if (cnt == 0) 1.0 else jac_sum / @as(f64, @floatFromInt(cnt));

    const mean = meanF64(fold_scores);
    const stdv = stdF64(fold_scores, mean);
    const runtime = @as(f64, @floatFromInt(timer.read())) / 1e9;

    return .{
        .method = method,
        .k = k,
        .cv_score_mean = mean,
        .cv_score_std = stdv,
        .stability_jaccard = stability,
        .runtime_sec = runtime,
    };
}

fn meanF64(v: []const f64) f64 {
    var s: f64 = 0;
    for (v) |x| s += x;
    return s / @as(f64, @floatFromInt(@max(v.len, 1)));
}

fn stdF64(v: []const f64, mean: f64) f64 {
    var s: f64 = 0;
    for (v) |x| s += (x - mean) * (x - mean);
    return std.math.sqrt(s / @as(f64, @floatFromInt(@max(v.len, 1))));
}

const Subset = struct {
    d: Dataset,
    owned_feature_names: [][]const u8,
    fn deinit(self: *Subset, alloc: Allocator) void {
        alloc.free(self.d.X);
        alloc.free(self.d.y);
        for (self.owned_feature_names) |s| alloc.free(s);
        alloc.free(self.owned_feature_names);
    }
};

fn makeSubsetDataset(alloc: Allocator, d: Dataset, idx: []const usize) !Subset {
    // subset rows only; keep feature names
    var X = try alloc.alloc(f64, idx.len * d.p);
    var y = try alloc.alloc(f64, idx.len);
    var i: usize = 0;
    while (i < idx.len) : (i += 1) {
        const ii = idx[i];
        y[i] = d.y[ii];
        const row = d.X[ii * d.p .. (ii + 1) * d.p];
        @memcpy(X[i * d.p .. (i + 1) * d.p], row);
    }
    var fnames = try alloc.alloc([]const u8, d.p);
    for (d.feature_names, 0..) |s, j| fnames[j] = try alloc.dupe(u8, s);
    return .{
        .d = Dataset{
            .X = X,
            .y = y,
            .feature_names = fnames,
            .n = idx.len,
            .p = d.p,
            .y_unique = d.y_unique,
            .y_is_int = d.y_is_int,
            .missing_rate = d.missing_rate,
            .task = d.task,
        },
        .owned_feature_names = fnames,
    };
}

fn selectBest(evals: []const MethodEval) MethodEval {
    var best = evals[0];
    for (evals[1..]) |e| {
        const better =
            (e.cv_score_mean > best.cv_score_mean) or
            (e.cv_score_mean == best.cv_score_mean and e.stability_jaccard > best.stability_jaccard) or
            (e.cv_score_mean == best.cv_score_mean and e.stability_jaccard == best.stability_jaccard and e.runtime_sec < best.runtime_sec);
        if (better) best = e;
    }
    return best;
}

fn writeExperienceJsonl(
    alloc: Allocator,
    store_path: []const u8,
    dataset_id: []const u8,
    meta: MetaFeatures,
    evals: []const MethodEval,
    best: MethodEval,
) !void {
    var file = try std.fs.cwd().openFile(store_path, .{ .mode = .write_only });
    defer file.close();
    try file.seekFromEnd(0);

    var buf = std.ArrayList(u8).init(alloc);
    defer buf.deinit();

    // minimal JSON (string builder)
    try buf.writer().print("{{\"dataset_id\":\"{s}\",", .{dataset_id});
    try buf.writer().print("\"meta_features\":{{\"n_samples\":{},\"n_features\":{},\"p_over_n\":{d:.6},\"missing_rate\":{d:.6},\"y_unique\":{},\"approx_task\":\"{s}\"}},", .{
        meta.n_samples,
        meta.n_features,
        meta.p_over_n,
        meta.missing_rate,
        meta.y_unique,
        if (meta.approx_task == .classification) "classification" else "regression",
    });
    try buf.writer().print("\"trajectory_features\":null,", .{});
    try buf.writer().print("\"evaluations\":{{", .{});
    for (evals, 0..) |e, i| {
        if (i != 0) try buf.appendSlice(",");
        try buf.writer().print("\"{s}\":{{\"method_name\":\"{s}\",\"k\":{},\"cv_score_mean\":{d:.12},\"cv_score_std\":{d:.12},\"stability_jaccard\":{d:.12},\"runtime_sec\":{d:.6}}}", .{
            methodName(e.method),
            methodName(e.method),
            e.k,
            e.cv_score_mean,
            e.cv_score_std,
            e.stability_jaccard,
            e.runtime_sec,
        });
    }
    try buf.writer().print("}},", .{});
    try buf.writer().print("\"selected_method\":\"{s}\",", .{methodName(best.method)});
    try buf.writer().print("\"selection_reason\":{{\"rule\":\"max(cv_score_mean)->max(stability)->min(runtime)\",\"best_metrics\":{{\"method_name\":\"{s}\",\"k\":{},\"cv_score_mean\":{d:.12},\"cv_score_std\":{d:.12},\"stability_jaccard\":{d:.12},\"runtime_sec\":{d:.6}}}}},", .{
        methodName(best.method),
        best.k,
        best.cv_score_mean,
        best.cv_score_std,
        best.stability_jaccard,
        best.runtime_sec,
    });
    // created_at omitted for simplicity in Zig version
    try buf.writer().print("\"created_at_utc\":null}}", .{});
    try buf.appendSlice("\n");

    _ = try file.writeAll(buf.items);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const args = parseArgs(alloc) catch |err| {
        if (err == error.Help or err == error.InvalidArgs) {
            printUsage();
            return;
        }
        return err;
    };

    const d = try loadCsv(alloc, args.csv_path.?, args.target.?);
    defer {
        alloc.free(d.X);
        alloc.free(d.y);
        for (d.feature_names) |s| alloc.free(s);
        alloc.free(d.feature_names);
    }

    const meta = MetaFeatures{
        .n_samples = d.n,
        .n_features = d.p,
        .p_over_n = @as(f64, @floatFromInt(d.p)) / @as(f64, @floatFromInt(@max(d.n, 1))),
        .missing_rate = d.missing_rate,
        .y_unique = d.y_unique,
        .approx_task = d.task,
    };

    var methods_list = std.ArrayList(Method).init(alloc);
    defer methods_list.deinit();
    if (args.methods.pearson) try methods_list.append(.pearson);
    if (args.methods.spearman) try methods_list.append(.spearman);
    if (args.methods.mi) try methods_list.append(.mi);
    if (methods_list.items.len == 0) return error.InvalidArgs;

    var evals = try alloc.alloc(MethodEval, methods_list.items.len);
    defer alloc.free(evals);

    for (methods_list.items, 0..) |m, i| {
        evals[i] = try evaluateMethod(alloc, d, m, args.k, args.cv);
        std.debug.print("method={s} mean={d:.6} std={d:.6} stability={d:.3} runtime={d:.3}s\n", .{
            methodName(m),
            evals[i].cv_score_mean,
            evals[i].cv_score_std,
            evals[i].stability_jaccard,
            evals[i].runtime_sec,
        });
    }

    const best = selectBest(evals);
    std.debug.print("selected_method: {s}\n", .{methodName(best.method)});

    const dataset_id = "zig_dataset";
    try writeExperienceJsonl(alloc, args.store, dataset_id, meta, evals, best);
    std.debug.print("experience_appended_to: {s}\n", .{args.store});
}

