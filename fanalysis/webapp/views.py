# myapp/views.py
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.generic import TemplateView

from .Utility.data_preprocessing import DataPreprocessing


class Home(TemplateView):
    template_name = 'home.html'


data_preprocessing_obj = ""
rfc, acc, prec, rec, f1, ad_prec, ad_rec,ad_f1, score_train, score_test = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
isUpload,isSettings,isTrain,isAttack, isDefence = False, False, False, False, False


def upload(request):
    context = {}

    global data_preprocessing_obj
    global isUpload, isSettings,isTrain,isAttack
    global rfc, acc, prec, rec, f1, ad_prec, ad_rec,ad_f1, score_train, score_test

    if request.method == 'POST':

        if 'btnUpload' in request.POST:
            uploaded_file = request.FILES['document']
            fs = FileSystemStorage()
            name = fs.save(uploaded_file.name, uploaded_file)
            context['url'] = fs.url(name)

            data_preprocessing_obj = DataPreprocessing(context['url'])
            context['columns'] = data_preprocessing_obj.read_file()
            isUpload = True
            context['isUpload'] = isUpload

    elif request.method == "GET":

        if 'btnSettings' in request.GET:

            label_parameter = request.GET['parameterSelect']

            model_type = request.GET['modelTypeSelect']

            data_preprocessing_obj.selectedParameter = label_parameter
            data_preprocessing_obj.selectedTrainingModel = model_type

            data_preprocessing_obj.filter_data()
            outlier, fraud, normal = data_preprocessing_obj.data_frame_describe()

            data_preprocessing_obj.implement_feature_selection()
            rfc, acc, prec, rec, f1 = data_preprocessing_obj.train_model()

            context['acc'] = round((acc * 100), 2)
            context['prec'] = round((prec * 100), 2)
            context['rec'] = round((rec * 100), 2)
            context['f1'] = round((f1 * 100), 2)
            isSettings = True
            context['isUpload'] = isUpload
            context['isSettings'] = isSettings

        elif "btnAttack" in request.GET:

            context['acc'] = round((acc * 100), 2)
            context['prec'] = round((prec * 100), 2)
            context['rec'] = round((rec * 100), 2)
            context['f1'] = round((f1 * 100), 2)
            context['isUpload'] = isUpload
            context['isSettings'] = isSettings

            percentage = int(request.GET.get('amount', '100'))
            attack_type = request.GET['attackTypeSelect']
            data_preprocessing_obj.dataPercentage = percentage / 100
            data_preprocessing_obj.selectedAttackType = attack_type

            print("tap attack button")

            score_train, score_test, ad_prec, ad_rec, ad_f1 = data_preprocessing_obj.attack()

            context['score_train'] = round((score_train * 100), 2)
            context['score_test'] = round((score_test * 100), 2)
            context['a_prec'] = round((ad_prec * 100), 2)
            context['a_rec'] = round((ad_rec * 100), 2)
            context['a_f1'] = round((ad_f1 * 100), 2)

            isAttack = True
            context['isAttack'] = isAttack

        elif "btnDefence" in request.GET:

            context['acc'] = round((acc * 100), 2)
            context['prec'] = round((prec * 100), 2)
            context['rec'] = round((rec * 100), 2)
            context['f1'] = round((f1 * 100), 2)

            context['a_prec'] = round((ad_prec * 100), 2)
            context['a_rec'] = round((ad_rec * 100), 2)
            context['a_f1'] = round((ad_f1 * 100), 2)
            context['isUpload'] = isUpload
            context['isAttack'] = isAttack
            context['isSettings'] = isSettings

            print("tap attack defence")

            context['score_train'] = round((score_train * 100), 2)
            context['score_test'] = round((score_test * 100), 2)

            latest_acc,  prec, rec, f1 = data_preprocessing_obj.defence()
            context['latest_acc'] = round((latest_acc * 100), 2)
            context['ad_prec'] = round((prec * 100), 2)
            context['ad_rec'] = round((rec * 100), 2)
            context['ad_f1'] = round((f1 * 100), 2)
            isDefence = True
            context['isDefence'] = isDefence

    return render(request, 'upload.html', context)
