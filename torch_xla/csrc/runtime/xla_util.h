#ifndef XLA_CLIENT_XLA_UTIL_H_
#define XLA_CLIENT_XLA_UTIL_H_

#include <string>

#include "absl/types/span.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/status_macros.h"
#include "torch_xla/csrc/runtime/types.h"

namespace torch_xla {
namespace runtime {
namespace util {

// Creates the HLO module which is generated by the input PB message.
xla::StatusOr<std::unique_ptr<xla::HloModule>> CreateModuleFromProto(
    const xla::HloModuleProto& proto,
    const xla::DebugOptions& debug_options = xla::DebugOptions());

// Returns a textual representation of the input XLA computation.
xla::StatusOr<std::string> GetComputationHloText(
    const xla::XlaComputation& computation);

void ReportComputationError(
    const xla::Status& status,
    absl::Span<const xla::XlaComputation* const> computations,
    absl::Span<const xla::Shape* const> output_shapes);

// Checks whether an action on the given computation generated an error, and if
// that was the case, emit error and computations HLO text.
void CheckComputationStatus(
    const xla::Status& status,
    absl::Span<const xla::XlaComputation* const> computations,
    absl::Span<const xla::Shape* const> output_shapes);

hash_t ShapeHash(const xla::Shape& shape);

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_XLA_UTIL_H_
