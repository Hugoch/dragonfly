// Copyright 2025, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "core/cms.h"
#include "error.h"
#include "facade/cmd_arg_parser.h"
#include "facade/error.h"
#include "facade/reply_builder.h"
#include "server/acl/acl_commands_def.h"
#include "server/command_registry.h"
#include "server/conn_context.h"
#include "server/engine_shard_set.h"
#include "server/transaction.h"

namespace dfly {

using namespace facade;
using namespace std;

namespace {

constexpr char kCmsNotFound[] = "CMS: key does not exist";
constexpr char kCmsWrongNumKeys[] = "CMS: wrong number of keys";
constexpr char kCmsWrongNumKeysWeights[] = "CMS: wrong number of keys/weights";
constexpr char kCmsCannotParseNumber[] = "CMS: Cannot parse number";

OpStatus OpInitByDim(const OpArgs& op_args, string_view key, uint32_t width, uint32_t depth) {
  auto& db_slice = op_args.GetDbSlice();
  auto op_res = db_slice.AddOrFind(op_args.db_cntx, key, OBJ_CMS);
  RETURN_ON_BAD_STATUS(op_res);

  if (!op_res->is_new)
    return OpStatus::KEY_EXISTS;

  PrimeValue& pv = op_res->it->second;
  pv.SetCMS(width, depth);

  return OpStatus::OK;
}

OpStatus OpInitByProb(const OpArgs& op_args, string_view key, double error, double probability) {
  auto& db_slice = op_args.GetDbSlice();
  auto op_res = db_slice.AddOrFind(op_args.db_cntx, key, OBJ_CMS);
  RETURN_ON_BAD_STATUS(op_res);

  if (!op_res->is_new)
    return OpStatus::KEY_EXISTS;

  PrimeValue& pv = op_res->it->second;
  CMS* cms = CompactObj::AllocateMR<CMS>(CMS::CreateByProb(error, probability,
                                                           CompactObj::memory_resource()));
  pv.SetCMS(cms);

  return OpStatus::OK;
}

OpResult<vector<int64_t>> OpIncrBy(const OpArgs& op_args, string_view key,
                                   const vector<pair<string_view, int64_t>>& items) {
  auto& db_slice = op_args.GetDbSlice();
  OpResult op_res = db_slice.FindMutable(op_args.db_cntx, key, OBJ_CMS);
  if (!op_res)
    return op_res.status();

  CMS* cms = op_res->it->second.GetCMS();
  vector<int64_t> result;
  result.reserve(items.size());

  for (const auto& [item, incr] : items) {
    result.push_back(cms->IncrBy(item, incr));
  }

  return result;
}

OpResult<vector<int64_t>> OpQuery(const OpArgs& op_args, string_view key, CmdArgList items) {
  auto& db_slice = op_args.GetDbSlice();
  OpResult op_res = db_slice.FindReadOnly(op_args.db_cntx, key, OBJ_CMS);
  if (!op_res)
    return op_res.status();

  const CMS* cms = op_res.value()->second.GetCMS();
  vector<int64_t> result;
  result.reserve(items.size());

  for (auto arg : items) {
    result.push_back(cms->Query(ToSV(arg)));
  }

  return result;
}

struct CmsInfo {
  uint32_t width;
  uint32_t depth;
  int64_t count;
};

OpResult<CmsInfo> OpInfo(const OpArgs& op_args, string_view key) {
  auto& db_slice = op_args.GetDbSlice();
  OpResult op_res = db_slice.FindReadOnly(op_args.db_cntx, key, OBJ_CMS);
  if (!op_res)
    return op_res.status();

  const CMS* cms = op_res.value()->second.GetCMS();
  return CmsInfo{cms->Width(), cms->Depth(), cms->Count()};
}

void CmdInitByDim(CmdArgList args, CommandContext* cmd_cntx) {
  CmdArgParser parser(args);
  string_view key = parser.Next();
  uint32_t width, depth;

  tie(width, depth) = parser.Next<uint32_t, uint32_t>();
  auto* rb = static_cast<RedisReplyBuilder*>(cmd_cntx->rb());
  RETURN_ON_PARSE_ERROR(parser, rb);

  if (width == 0 || depth == 0) {
    return rb->SendError("CMS: width and depth must be greater than 0");
  }

  const auto cb = [&](Transaction* t, EngineShard* shard) {
    return OpInitByDim(t->GetOpArgs(shard), key, width, depth);
  };

  OpStatus res = cmd_cntx->tx()->ScheduleSingleHop(std::move(cb));
  if (res == OpStatus::KEY_EXISTS) {
    return rb->SendError("item exists");
  }
  if (res == OpStatus::OK) {
    return rb->SendOk();
  }
  return rb->SendError(res);
}

void CmdInitByProb(CmdArgList args, CommandContext* cmd_cntx) {
  CmdArgParser parser(args);
  string_view key = parser.Next();
  double error, probability;

  tie(error, probability) = parser.Next<double, double>();
  auto* rb = static_cast<RedisReplyBuilder*>(cmd_cntx->rb());
  RETURN_ON_PARSE_ERROR(parser, rb);

  if (error <= 0 || error >= 1) {
    return rb->SendError("CMS: error must be between 0 and 1 exclusive");
  }
  if (probability <= 0 || probability >= 1) {
    return rb->SendError("CMS: probability must be between 0 and 1 exclusive");
  }

  const auto cb = [&](Transaction* t, EngineShard* shard) {
    return OpInitByProb(t->GetOpArgs(shard), key, error, probability);
  };

  OpStatus res = cmd_cntx->tx()->ScheduleSingleHop(std::move(cb));
  if (res == OpStatus::KEY_EXISTS) {
    return rb->SendError("item exists");
  }
  if (res == OpStatus::OK) {
    return rb->SendOk();
  }
  return rb->SendError(res);
}

void CmdIncrBy(CmdArgList args, CommandContext* cmd_cntx) {
  string_view key = ArgS(args, 0);
  args.remove_prefix(1);

  // Parse item/increment pairs
  if (args.size() < 2 || args.size() % 2 != 0) {
    return cmd_cntx->SendError(kSyntaxErr);
  }

  vector<pair<string_view, int64_t>> items;
  items.reserve(args.size() / 2);

  for (size_t i = 0; i < args.size(); i += 2) {
    string_view item = ToSV(args[i]);
    int64_t incr;
    if (!absl::SimpleAtoi(ToSV(args[i + 1]), &incr)) {
      return cmd_cntx->SendError(kCmsCannotParseNumber);
    }
    items.emplace_back(item, incr);
  }

  const auto cb = [&](Transaction* t, EngineShard* shard) {
    return OpIncrBy(t->GetOpArgs(shard), key, items);
  };

  auto* rb = static_cast<RedisReplyBuilder*>(cmd_cntx->rb());
  OpResult<vector<int64_t>> res = cmd_cntx->tx()->ScheduleSingleHopT(std::move(cb));
  if (!res) {
    if (res.status() == OpStatus::KEY_NOTFOUND) {
      return rb->SendError(kCmsNotFound);
    }
    return rb->SendError(res.status());
  }

  SinkReplyBuilder::ReplyScope scope(rb);
  rb->StartArray(res->size());
  for (int64_t count : *res) {
    rb->SendLong(count);
  }
}

void CmdQuery(CmdArgList args, CommandContext* cmd_cntx) {
  string_view key = ArgS(args, 0);
  args.remove_prefix(1);

  if (args.empty()) {
    return cmd_cntx->SendError(kSyntaxErr);
  }

  const auto cb = [&](Transaction* t, EngineShard* shard) {
    return OpQuery(t->GetOpArgs(shard), key, args);
  };

  auto* rb = static_cast<RedisReplyBuilder*>(cmd_cntx->rb());
  OpResult<vector<int64_t>> res = cmd_cntx->tx()->ScheduleSingleHopT(std::move(cb));
  if (!res) {
    if (res.status() == OpStatus::KEY_NOTFOUND) {
      return rb->SendError(kCmsNotFound);
    }
    return rb->SendError(res.status());
  }

  SinkReplyBuilder::ReplyScope scope(rb);
  rb->StartArray(res->size());
  for (int64_t count : *res) {
    rb->SendLong(count);
  }
}

void CmdInfo(CmdArgList args, CommandContext* cmd_cntx) {
  string_view key = ArgS(args, 0);

  auto* rb = static_cast<RedisReplyBuilder*>(cmd_cntx->rb());

  const auto cb = [&](Transaction* t, EngineShard* shard) {
    return OpInfo(t->GetOpArgs(shard), key);
  };

  OpResult<CmsInfo> res = cmd_cntx->tx()->ScheduleSingleHopT(std::move(cb));
  if (!res) {
    if (res.status() == OpStatus::KEY_NOTFOUND) {
      return rb->SendError(kCmsNotFound);
    }
    return rb->SendError(res.status());
  }

  rb->StartArray(6);
  rb->SendBulkString("width");
  rb->SendLong(res->width);
  rb->SendBulkString("depth");
  rb->SendLong(res->depth);
  rb->SendBulkString("count");
  rb->SendLong(res->count);
}

void CmdMerge(CmdArgList args, CommandContext* cmd_cntx) {
  CmdArgParser parser(args);
  string_view dest_key = parser.Next();
  uint32_t num_keys;

  num_keys = parser.Next<uint32_t>();
  auto* rb = static_cast<RedisReplyBuilder*>(cmd_cntx->rb());
  RETURN_ON_PARSE_ERROR(parser, rb);

  if (num_keys == 0) {
    return rb->SendError(kCmsWrongNumKeys);
  }

  // Check if we have enough arguments for keys
  if (parser.Tail().size() < num_keys) {
    return rb->SendError(kSyntaxErr);
  }

  vector<string_view> src_keys;
  src_keys.reserve(num_keys);
  for (uint32_t i = 0; i < num_keys; ++i) {
    src_keys.push_back(parser.Next());
  }

  // Parse optional WEIGHTS
  vector<int64_t> weights;
  if (parser.HasNext()) {
    string_view weights_kw = parser.Next();
    if (!absl::EqualsIgnoreCase(weights_kw, "WEIGHTS")) {
      return rb->SendError(kSyntaxErr);
    }

    weights.reserve(num_keys);
    for (uint32_t i = 0; i < num_keys; ++i) {
      if (!parser.HasNext()) {
        return rb->SendError(kCmsWrongNumKeysWeights);
      }
      int64_t w;
      if (!absl::SimpleAtoi(parser.Next(), &w)) {
        return rb->SendError(kCmsCannotParseNumber);
      }
      weights.push_back(w);
    }
  }

  if (parser.HasNext()) {
    return rb->SendError(kCmsWrongNumKeysWeights);
  }

  // Use default weights of 1 if not specified
  if (weights.empty()) {
    weights.resize(num_keys, 1);
  }

  // For CMS.MERGE, we need to check that all keys exist and have matching dimensions
  // Then merge source CMS into destination
  // This requires coordination across shards if keys are distributed

  // For now, use a simple approach: read sources, then merge into destination
  // This works when keys are on same shard (ScheduleSingleHop will fail if not)
  auto cb = [&](Transaction* t, EngineShard* shard) -> OpStatus {
    auto& db_slice = t->GetOpArgs(shard).GetDbSlice();
    const DbContext& db_cntx = t->GetDbContext();

    // Find destination
    LOG(ERROR) << "Find destination";
    OpResult dest_res = db_slice.FindMutable(db_cntx, dest_key, OBJ_CMS);
    if (!dest_res) {
      LOG(ERROR) << "Destination does not exist";
      return dest_res.status();
    }
    CMS* dest_cms = dest_res->it->second.GetCMS();

    // Read all source CMS and merge
    for (size_t i = 0; i < src_keys.size(); ++i) {
      LOG(ERROR) << "Find source";
      OpResult src_res = db_slice.FindReadOnly(db_cntx, src_keys[i], OBJ_CMS);
      if (!src_res) {
        LOG(ERROR) << "Source does not exist" << " " << src_keys[i];
        return src_res.status();
      }
      const CMS* src_cms = src_res.value()->second.GetCMS();

      if (!dest_cms->MergeFrom(*src_cms, weights[i])) {
        return OpStatus::INVALID_VALUE;  // Dimension mismatch
      }
    }

    return OpStatus::OK;
  };

  OpStatus res = cmd_cntx->tx()->ScheduleSingleHop(std::move(cb));
  if (res == OpStatus::KEY_NOTFOUND) {
    return rb->SendError(kCmsNotFound);
  }
  if (res == OpStatus::INVALID_VALUE) {
    return rb->SendError("CMS: dimension mismatch");
  }
  if (res == OpStatus::OK) {
    return rb->SendOk();
  }
  return rb->SendError(res);
}

}  // namespace

using CI = CommandId;

#define HFUNC(x) SetHandler(&Cmd##x)

void RegisterCmsFamily(CommandRegistry* registry) {
  registry->StartFamily();

  *registry
      << CI{"CMS.INITBYDIM", CO::JOURNALED | CO::DENYOOM | CO::FAST, 4, 1, 1, acl::BLOOM}.HFUNC(
             InitByDim)
      << CI{"CMS.INITBYPROB", CO::JOURNALED | CO::DENYOOM | CO::FAST, 4, 1, 1, acl::BLOOM}.HFUNC(
             InitByProb)
      << CI{"CMS.INCRBY", CO::JOURNALED | CO::DENYOOM | CO::FAST, -4, 1, 1, acl::BLOOM}.HFUNC(IncrBy)
      << CI{"CMS.QUERY", CO::READONLY | CO::FAST, -3, 1, 1, acl::BLOOM}.HFUNC(Query)
      << CI{"CMS.INFO", CO::READONLY | CO::FAST, 2, 1, 1, acl::BLOOM}.HFUNC(Info)
      << CI{"CMS.MERGE", CO::JOURNALED | CO::DENYOOM, -4, 1, 1, acl::BLOOM}.HFUNC(Merge);
}

}  // namespace dfly
