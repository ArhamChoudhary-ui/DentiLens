export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    throw new Error(typeof error?.detail === 'string' ? error.detail : `Request failed (${response.status}).`)
  }
  return response.json() as Promise<T>
}

export interface UploadProgress {
  percent: number
  loaded: number
  total: number
}

export function uploadWithProgress<T>(
  path: string,
  formData: FormData,
  onProgress?: (progress: UploadProgress) => void,
  abortSignal?: AbortSignal
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', path)
    xhr.responseType = 'json'

    if (abortSignal) {
      if (abortSignal.aborted) {
        reject(new DOMException('Upload cancelled', 'AbortError'))
        return
      }
      abortSignal.addEventListener('abort', () => {
        xhr.abort()
        reject(new DOMException('Upload cancelled', 'AbortError'))
      })
    }

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (event: ProgressEvent) => {
        if (event.lengthComputable && event.total > 0) {
          const percent = Math.min(100, Math.round((event.loaded / event.total) * 100))
          onProgress({ percent, loaded: event.loaded, total: event.total })
        }
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as T)
      } else {
        const error = xhr.response
        const detail = typeof error?.detail === 'string' ? error.detail : `Upload failed (${xhr.status}).`
        reject(new Error(detail))
      }
    }

    xhr.onerror = () => {
      reject(new Error('Network error during upload. Please check your connection and retry.'))
    }

    xhr.ontimeout = () => {
      reject(new Error('Upload timed out. Please retry.'))
    }

    xhr.send(formData)
  })
}
