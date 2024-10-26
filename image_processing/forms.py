from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

class SmoothingForm(forms.Form):
    mask_size = forms.IntegerField(label='Mask Size', min_value=1, max_value=10)


class BoxForm(forms.Form):
    mask_size = forms.IntegerField(label='Mask Size', min_value=1, max_value=10)


class DogForm(forms.Form):
    mask_size1 = forms.IntegerField(label='Mask Size 1', min_value=1, max_value=10)
    mask_size2 = forms.IntegerField(label='Mask Size 2', min_value=1, max_value=10)


from django import forms

class MultiImageUploadForm(forms.Form):
    images = forms.FileField(widget=forms.ClearableFileInput(attrs={'allow_multiple_selected': True}))

    



