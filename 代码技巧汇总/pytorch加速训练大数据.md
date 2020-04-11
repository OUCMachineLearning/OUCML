## Training Faster With Large Datasets using Scale and PyTorch

Authored by [Daniel Havir](https://www.linkedin.com/in/danielhavir/) & [Nathan Hayflick](https://www.linkedin.com/in/nathan-hayflick-8b1a1435/) at Scale AI.

*Scale AI, the Data Platform for AI development, shares some tips on how ML engineers can more easily build and work with large datasets by using PyTorch’s asynchronous data loading capabilities and Scale as a continuous labeling pipeline.*

When training models for production applications, ML engineers often need to iterate on their training datasets as much as they iterate on their model architecture. This is because increasing the size, variety or labeling quality of a dataset is one of our most powerful levers to increase model performance¹. Ideally our machine learning pipeline should make acquiring new data and retraining as quick and seamless as possible to maximize these performance gains. However engineers face several barriers to retraining on larger datasets including the complexity of managing large-scale data labeling operations and the time (as well as compute cost) of retraining on more data. This article demonstrates some techniques for managing the kinds of issues specific to working with large datasets by using Scale’s labeling platform and PyTorch’s rich support for custom training workflows.

## Data Labeling and Annotation

The first challenge to training on larger datasets is simply obtaining large quantities of annotations without sacrificing quality. While smaller datasets can be labeled successfully one-off by a few labelers (or even ML engineers themselves), building datasets composed of hundreds of thousands of scenes require a large amount of QA processes and tooling. Ensuring labeling rules are followed consistently across a large and diverse group of labelers is key for improving accuracy in training. Scale’s Labeling API was designed to abstract away many of these concerns so whether you are collecting a small prototype training set or millions of labels the process is as painless as possible.

We start with a collection of unlabeled images of indoor scenes that we’ve collected for our project, like the following example:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-10-152119.png)

*Example indoor scene for segmentation*

In this example, we show how to train a model to identify pieces of furniture, appliances and architectural features on increasingly larger datasets to improve model performance. This model is similar to the types of computer vision solutions we see being deployed in retail, real estate or e-commerce. We start by establishing in what format we need our annotations. Our labels must match the desired output format for our model. While there are several different common output formats for computer vision models (including boxes and polygons), for this task, we’ve chosen to use a semantic segmentation model. This format allows our model to provide the greatest amount of environmental context — context that comes at the expense of greater computational resources (more on this later).

Scale provides a [Semantic Segmentation endpoint](https://scale.com/docs#semantic-segmentation) where we can submit our unlabeled images along with labeling instructions and our desired labeling structure.

```
import scaleapi

client = scaleapi.ScaleClient('YOUR_SCALE_API_KEY')

client.create_segmentannotation_task(
  callback_url='http://www.example.com/callback',
  instruction='Please segment the image using the given labels',
  attachment_type='image',
  attachment='https://live.staticflickr.com/3810/9200809507_ba231b5f9b_b.jpg',
  labels=[
    {
        'choice': 'furniture',
        'subchoices': ['table', 'sofa', 'chair', 'shelf', 'wardrobe', 'stand', 'cabinet', 'ottoman/foot-rest', 'rug']
    },
    {
        'choice': 'lighting',
        'subchoices': ['floorlamp', 'tablelamp', 'ceiling fixture', 'chandelier', 'sconce']
      },
      {
        'choice': 'architectural feature',
        'subchoices': ['stairs', 'window', 'wall', 'floor', 'ceiling', 'fireplace']
      },
      {
        'choice': 'electronic devices',
        'subchoices': ['speakers', 'television', 'computer', 'clock', 'other device']
      },
      {
        'choice': 'containers',
        'subchoices': ['trashcan', 'basket', 'box', 'vase']
      },
      {
        'choice': 'personal items',
        'subchoices': ['toy', 'book', 'pillow']
      },
      {
        'choice': 'decoration',
        'subchoices': ['painting/picture', 'plant', 'other decoration']
      }
    ],
  allow_unlabeled=False
)
```

Our label mapping hierarchy must encompass all the classes of objects we are interested in predicting, such as furniture, lighting, etc. While Scale supports generating panoptic labels (where we store a separate pixel map for each distinct object within a class), we will be using a classic semantic segmentation approach here.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-10-152207.png)

*Data Labeling Pipeline*

Once Scale has completed the labeling process, a task response will be sent to our designated callback url with links to a set of PNGs. These PNGS contain the combined, indexed and individual labeling layers that we can use to train our model in the following steps.



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-10-152212.png)

*Scene segmentation — each color represents a label layer.*

## Getting Started With Local Training

Now that we are receiving data from our labeling pipeline, we can train a prototype model with a simple implementation that we can run locally or on a single notebook instance. We want to start with a smaller subset of our dataset to speed prototyping and benchmark performance before increasing the quantity of training data in later steps.

By setting up our ML environment with PyTorch, we can leverage an ecosystem of popular model architectures to start training on. Here we import the DeeplabV3 model from the TorchVision package and then perform some pre-processing on our Scale formatted labeled validation set before loading data into the model.

```
# Full Example: https://gist.github.com/danielhavir/407a6cfd592dfc2ad1e23a1ed3539e07
import os
from typing import Callable, List, Tuple, Generator, Dict

import torch
import torch.utils.data
from PIL.Image import Image as ImageType


def list_items_local(path: str) -> List[str]:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(path))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, items: List[str], loader: Callable[[str], ImageType] = pil_loader, transform=None):
        self.data_root = data_root
        self.loader = loader
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        item_id = self.items[item]

        image = self.loader(os.path.join(self.data_root, "images", item_id + ".jpg"))
        label = self.loader(os.path.join(self.data_root, "labels", item_id + ".png"))

        if self.transform is not None:
            image, label = self.transform((image, label))

        return image, label


def get_local_dataloaders(local_data_root: str, batch_size: int = 8, transform: Callable = None,
                          test_ratio: float = 0.1, num_workers: int = 8) -> Dict[str, torch.utils.data.DataLoader]:
    # Local training
    local_items = list_items_local(os.path.join(local_data_root, "images"))
    dataset = ImageDataset(local_data_root, local_items, transform=transform)

    # Split using consistent hashing
    train_indices, test_indices = consistent_train_test_split(local_items, test_ratio)
    return {
        "train": torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices),
                                             num_workers=num_workers),
        "test": torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_indices),
                                            num_workers=num_workers)
    }
```

One thing to keep in mind, your production dataset will change over time as more data is labeled, so it is important to keep track of your train/test split in order to evaluate your models consistently on the same examples. An easy way to achieve a consistent split is to use a hashing function. This guarantees that all of your test examples remain in your test set even while your dataset continues to grow.

Now we can run our validation tests and confirm the model’s training is working!

```
from zlib import crc32

def test_set_check(identifier: str, test_ratio: float) -> bool:
    # Source: https://datascience.stackexchange.com/questions/51348/splitting-train-test-sets-by-an-identifier 
    hashcode = float(crc32(bytes(identifier, "utf-8")) & 0xffffffff) / (1 << 32)
    return hashcode < test_ratio
```

## Streaming Data With PyTorch

At this point, it’s common to begin diving into the architecture of the model or fine-tuning. Instead, we are going to take a different approach and see how far we can get by simply increasing the size of our training dataset. One challenge we will have to address is that our training code is currently slow and memory intensive due to the size requirements of our segmentation data (each labeled scene is a full-size PNG). Spinning up a development environment also becomes increasingly painful if we are required to preload gigabytes of data before running. To continue to expand the size of our dataset we will need to modify our approach to work outside the bounds of a single machine.

The flexibility of PyTorch’s DataLoader class enables us to implement data streaming fairly easily by customizing how our Dataset class loads items. This allows us to move our dataset from disk to cloud storage (GCS or the like) and progressively stream our labels as we train.

```
import io
from urllib.parse import urlparse

from PIL import Image
from google.cloud import storage
from google.api_core.retry import Retry


@Retry()
def gcs_pil_loader(uri: str) -> ImageType:
    uri = urlparse(uri)
    client = storage.Client()
    bucket = client.get_bucket(uri.netloc)
    b = bucket.blob(uri.path[1:], chunk_size=None)
    image = Image.open(io.BytesIO(b.download_as_string()))
    return image.convert("RGB")


@Retry()
def load_items_gcs(path: str) -> List[str]:
    uri = urlparse(path)
    client = storage.Client()
    bucket = client.get_bucket(uri.netloc)
    blobs = bucket.list_blobs(prefix=uri.path[1:], delimiter=None)
    return sorted(os.path.splitext(os.path.basename(blob.name))[0] for blob in blobs)


def get_streamed_dataloaders(gcs_data_root: str, batch_size: int = 8, transform: Callable = None,
                             test_ratio: float = 0.1, num_workers: int = 8) -> Dict[str, torch.utils.data.DataLoader]:
    # Streaming
    streamed_items = load_items_gcs(os.path.join(gcs_data_root, "images"))
    dataset = ImageDataset(gcs_data_root, streamed_items, loader=gcs_pil_loader, transform=transform)

    # Identical for both local and streamed
    # This is handy for CrossValidation, use consistent hashing
    train_indices, test_indices = consistent_train_test_split(streamed_items, test_ratio)
    return {
        "train": torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=torch.utils.data.SubsetRandomSampler(train_indices),
                                             num_workers=num_workers),
        "test": torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=torch.utils.data.SubsetRandomSampler(test_indices),
                                            num_workers=num_workers)
    }
```

When we need to run locally, we can use `pil_loader` as the loader function and then switch to `gcs_pil_loader` for streaming.

Moving to a streaming architecture also allows us to save on training costs by migrating to preemptible instances (Scale’s ML team uses Kubernetes which integrates nicely with preemptible instances on most cloud providers). This gives us access to cheaper, on-demand hardware as long as our workflows are fault-tolerant to instance shutdowns.

It also allows us to easily experiment with different samples of our dataset by interacting with our data as a set of indexed URLs in a storage bucket (preventing us from needing to transfer files in bulk everytime we want to change the distribution).

We can see this approach scales much better, allowing us to keep our data in one central place without needing to download massive datasets to an instance every time we start training — our benchmarks show an average data load time of 25.04ms. To speed things up here, we may want to consider an additional modification: asynchronous streaming.

## Asynchronous Streaming

To further ease working with a larger dataset, we can minimize data loading time by moving to an async workflow.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-10-152206.png)

*Asynchronous Training Pipeline*

In our previous streaming example, training speed was bottlenecked by the speed of each data request (which introduces significant latency when loading training data from a cloud object store such as GCS or S3). This latency can be negated using concurrency, i.e. latency hiding. We evaluated several options for concurrency in Python including multithreading, multiprocessing and asyncio. The Python libraries for both S3 and GCS block the global interpreter lock, so gains are small with multithreading. Multiprocessing gives us true concurrency — however this comes at the expense of memory usage and in our tests we frequently ran into memory-constraints when we tried to scale up the number of processes in production. In the end, we landed on asyncio as the best solution for highly-concurrent data loading. Asyncio is suitable for IO-bound and high-level structured network code. DataLoader already achieves some concurrency using PyTorch’s [multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html), however for the purpose of network latency hiding, we can achieve higher concurrency at a lower cost using asyncio. We use both multiprocessing and asyncio in production, with each process running its own asyncio event loop. Multiprocessing gives us concurrency of CPU-intensive tasks (such as conversion to torch tensor or data augmentation) while the event loops hide the latency of hundreds of GCS/S3 requests.

This requires some rewriting of our code to handle concurrency.

```
import random
import asyncio

import aiohttp
from janus import Queue
from gcloud.aio.storage import Storage


def generate_stream(items: List[str]) -> Generator[str, None, None]:
    while True:
        # Python's randint has inclusive upper bound
        index = random.randint(0, len(items) - 1)
        yield items[index]


class AsyncImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_root: str, items: List[str], transform: Callable = None, concurrency: int = 64):
        self.data_root = data_root
        self.items = items
        self.transform = transform
        self.worker_initialized = False
        self.loop_thread = None
        self.q = None
        self.creds = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        self.concurrency = concurrency
        self.stream = generate_stream(self.items)

    async def run(self, loop, session):
        for item in self.stream:
            try:
                image_gs = urlparse(os.path.join(self.data_root, "images", item + ".jpg"))
                label_gs = urlparse(os.path.join(self.data_root, "labels", item + ".png"))
                aio_storage = Storage(service_file=self.creds, session=session)
                blobs = await asyncio.gather(
                    aio_storage.download(image_gs.netloc, image_gs.path[1:]),
                    aio_storage.download(label_gs.netloc, label_gs.path[1:]),
                    loop=loop
                )
                image = Image.open(io.BytesIO(blobs[0]))
                label = Image.open(io.BytesIO(blobs[1])).convert("RGB")
                await self.q.async_q.put((image, label))
            except aiohttp.ClientError as e:
                logging.debug(e)
            except TimeoutError:
                pass
            except Exception as e:
                logging.exception(e)

    def init_worker(self):
        loop = asyncio.new_event_loop()
        session = aiohttp.ClientSession(loop=loop, connector=aiohttp.TCPConnector(limit=0, loop=loop),
                                        raise_for_status=True)
        self.q = Queue(self.concurrency, loop=loop)

        # Spin up workers
        for _ in range(self.concurrency):
            loop.create_task(self.run(loop, session))

        def loop_in_thread(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self.loop_thread = Thread(target=loop_in_thread, args=(loop,), daemon=True)
        self.loop_thread.start()
        self.worker_initialized = True

    def __iter__(self):
        while True:
            if not self.worker_initialized:
                self.init_worker()

            image, label = self.q.sync_q.get()
            if self.transform is not None:
                image, label = self.transform((image, label))

            yield image, label


def get_async_dataloaders(gcs_data_root: str, batch_size: int = 8, transform: Callable = None,
                          test_ratio: float = 0.1, num_workers: int = 8) -> Dict[str, torch.utils.data.DataLoader]:
    # Async Streaming
    streamed_items = load_items_gcs(os.path.join(gcs_data_root, "images"))
    train_indices, test_indices = consistent_train_test_split(streamed_items, test_ratio)
    train_items = [streamed_items[i.item()] for i in train_indices]
    train_dataset = AsyncImageDataset(gcs_data_root, train_items, transform=transform, concurrency=128)
    test_items = [streamed_items[i.item()] for i in test_indices]
    test_dataset = AsyncImageDataset(gcs_data_root, test_items, transform=transform, concurrency=128)

    return {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, worker_init_fn=worker_init_fn,
                                num_workers=num_workers),
        "test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, worker_init_fn=worker_init_fn,
                               num_workers=num_workers)
    }
```

Looking at some training benchmarks, we can see that synchronous streaming triggers an image loading bottleneck that causes the average batch time to spike to almost 20 seconds (when using the largest batch size), while the async streaming code took less than a second on average. Training becomes much more efficient since we are able to keep the GPUs saturated even as our batch sizes increase.

## Data loading benchmarks



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-10-152209.png)

Before finishing up, we will want to add a series of data augmentation techniques to artificially increase the size of our dataset even further. Some color jitter, random affine transforms and random cropping are all good starting points.

# Wrapping up: A system for constant dataset iteration

Through this article we’ve moved from a prototype tied to a small static dataset to a more flexible training system. We can now swap out and experiment with increasingly larger datasets (provided by our labeling pipeline with Scale) while minimizing retraining time and cost. In the future we can make our distributed training workflow even more efficient with the recently released [PyTorch Elastic](https://github.com/pytorch/elastic) framework which makes it even easier to work with a dynamic pool of spot instances. By integrating Scale’s labeling platform and PyTorch’s tooling for distributed training workflows, ML engineers have a powerful set of building blocks to continuously expand our selection of training data and repeatedly drive performance gains in their ML models.

## References

¹ https://developers.google.com/machine-learning/data-prep/construct/collect/data-size-quality