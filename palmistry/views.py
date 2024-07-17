from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
import torch
from .tools import remove_background, resize, save_result, print_error
from .model import UNet
from .rectification import warp
from .detection import detect
from .classification import classify
from .measurement import measure

@csrf_exempt
def detect_palm(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        # Save the uploaded image to a temporary location
        image_path = os.path.join(settings.BASE_DIR, 'temp_image.jpg')
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        results_dir = os.path.join(settings.BASE_DIR, 'results')
        os.makedirs(results_dir, exist_ok=True)

        resize_value = 256
        path_to_clean_image = os.path.join(results_dir, 'palm_without_background.jpg')
        path_to_warped_image = os.path.join(results_dir, 'warped_palm.jpg')
        path_to_warped_image_clean = os.path.join(results_dir, 'warped_palm_clean.jpg')
        path_to_warped_image_mini = os.path.join(results_dir, 'warped_palm_mini.jpg')
        path_to_warped_image_clean_mini = os.path.join(results_dir, 'warped_palm_clean_mini.jpg')
        path_to_palmline_image = os.path.join(results_dir, 'palm_lines.png')
        path_to_model = os.path.join(settings.BASE_DIR, 'palmistry', 'checkpoint', 'checkpoint_aug_epoch70.pth')
        path_to_result = os.path.join(results_dir, 'result.jpg')

        # 0. Preprocess image
        remove_background(image_path, path_to_clean_image)

        # 1. Palm image rectification
        warp_result = warp(image_path, path_to_warped_image)
        if warp_result is None:
            print_error()
            return JsonResponse({'error': 'Warping failed'})

        remove_background(path_to_warped_image, path_to_warped_image_clean)
        resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value)

        # 2. Principal line detection
        net = UNet(n_channels=3, n_classes=1)
        net.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        detect(net, path_to_warped_image_clean, path_to_palmline_image, resize_value)

        # 3. Line classification
        lines = classify(path_to_palmline_image)

        # 4. Length measurement
        im, contents = measure(path_to_warped_image_mini, lines)

        # 5. Save result
        save_result(im, contents, resize_value, path_to_result)

        heart_content_2, head_content_2, life_content_2, marriage_content_2, fate_content_2 = contents

        return JsonResponse({
            'heart_content_2': heart_content_2,
            'head_content_2': head_content_2,
            'life_content_2': life_content_2,
            'marriage_content_2': marriage_content_2,
            'fate_content_2': fate_content_2
        })

    return JsonResponse({'error': 'No image uploaded'})
