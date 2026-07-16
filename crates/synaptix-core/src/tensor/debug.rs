use std::fmt;

use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::tensor::storage::Storage;

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor {{ shape: {:?}, dtype: {:?}, device: {:?}, contiguous: {} }}",
            self.dims(),
            self.dtype(),
            self.device(),
            self.is_contiguous()
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Tensor[{:?} {:?} on {:?}]",
            self.dims(),
            self.dtype(),
            self.device()
        )?;
        if !matches!(&*self.storage, Storage::Cpu(_)) {
            return write!(f, "  <not on cpu; use .to_device(Cpu) to print>");
        }
        if !self.is_contiguous() {
            return write!(f, "  <non-contiguous>");
        }
        match self.dtype() {
            DType::F32 => print_typed::<f32>(self, f),
            DType::F64 => print_typed::<f64>(self, f),
            DType::F16 => print_typed::<half::f16>(self, f),
            DType::BF16 => print_typed::<half::bf16>(self, f),
            DType::U8 => print_typed::<u8>(self, f),
            DType::U32 => print_typed::<u32>(self, f),
            DType::I32 => print_typed::<i32>(self, f),
            DType::I64 => print_typed::<i64>(self, f),
            _ => write!(f, "  <printing not implemented for {:?}>", self.dtype()),
        }
    }
}

fn print_typed<T: crate::dtype::SynaptixScalar + fmt::Display>(
    t: &Tensor,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    const PER_LINE: usize = 8;
    const MAX_ELEMS: usize = 64;
    let host: Vec<T> = match t.to_host_for_debug::<T>() {
        Ok(v) => v,
        Err(_) => return write!(f, "  <error reading storage>"),
    };
    let total = host.len();
    let show = total.min(MAX_ELEMS);
    for (i, v) in host.iter().take(show).enumerate() {
        if i % PER_LINE == 0 {
            write!(f, "  ")?;
        }
        write!(f, "{:>8.4} ", v)?;
        if (i + 1) % PER_LINE == 0 {
            writeln!(f)?;
        }
    }
    if show < total {
        writeln!(f, "\n  ... ({} more)", total - show)?;
    } else if show % PER_LINE != 0 {
        writeln!(f)?;
    }
    Ok(())
}

impl Tensor {
    fn to_host_for_debug<T: crate::dtype::SynaptixScalar>(&self) -> crate::error::Result<Vec<T>> {
        let elem_bytes = std::mem::size_of::<T>();
        let offset_bytes = self.layout.offset() * elem_bytes;
        let numel = self.numel();
        match &*self.storage {
            Storage::Cpu(b) => {
                let bytes = &b.as_bytes()[offset_bytes..offset_bytes + numel * elem_bytes];
                Ok(bytemuck::cast_slice(bytes).to_vec())
            }
            _ => Err(crate::error::SynaptixError::Unsupported("debug print non-cpu")),
        }
    }
}
