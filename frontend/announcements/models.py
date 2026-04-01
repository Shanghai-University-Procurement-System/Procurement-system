from django.db import models
import datetime
from wagtail.models import Page
from wagtail.fields import RichTextField
from wagtail.admin.panels import FieldPanel
from wagtail.search import index
from django.core.paginator import Paginator
from django.contrib.auth.models import User


# 公告类型选择
ANNOUNCEMENT_TYPE_CHOICES = [
    ('winning_bid', '中标公告'),
    ('public_bidding', '公开招标公告'),
    ('competitive_negotiation', '竞争性谈判公告'),
    ('correction', '更正公告'),
    ('competitive_consultation', '竞争性磋商公告'),
    ('transaction', '成交公告'),
    ('other', '其他公告'),
    ('termination', '终止公告'),
    ('qualification', '资格预审公告'),
    ('inquiry', '询价公告'),
]

# 地区选择（省级）
REGION_CHOICES = [
    ('all', '全部'),
    ('beijing', '北京'),
    ('tianjin', '天津'),
    ('hebei', '河北'),
    ('shanxi', '山西'),
    ('neimenggu', '内蒙古'),
    ('liaoning', '辽宁'),
    ('jilin', '吉林'),
    ('heilongjiang', '黑龙江'),
    ('shanghai', '上海'),
    ('jiangsu', '江苏'),
    ('zhejiang', '浙江'),
    ('anhui', '安徽'),
    ('fujian', '福建'),
    ('jiangxi', '江西'),
    ('shandong', '山东'),
    ('henan', '河南'),
    ('hubei', '湖北'),
    ('hunan', '湖南'),
    ('guangdong', '广东'),
    ('guangxi', '广西'),
    ('hainan', '海南'),
    ('chongqing', '重庆'),
    ('sichuan', '四川'),
    ('guizhou', '贵州'),
    ('yunnan', '云南'),
    ('xizang', '西藏'),
    ('shaanxi', '陕西'),
    ('gansu', '甘肃'),
    ('qinghai', '青海'),
    ('ningxia', '宁夏'),
    ('xinjiang', '新疆'),
    ('hongkong', '香港'),
    ('macau', '澳门'),
    ('taiwan', '台湾'),
]


# 采购品目选择
PROCUREMENT_CATEGORY_CHOICES = [
    ('goods', '货物类'),
    ('engineering', '工程类'),
    ('service', '服务类'),
]

# 所属行业选择
INDUSTRY_CHOICES = [
    ('construction', '建筑工程'),
    ('water', '水利水电'),
    ('energy', '能源化工'),
    ('security', '弱电安防'),
    ('it', '信息技术'),
    ('office', '行政办公'),
    ('machinery', '机械设备'),
    ('transportation', '交通工程'),
    ('medical', '医疗卫生'),
    ('municipal', '市政设施'),
    ('service', '服务采购'),
    ('agriculture', '农林牧渔'),
    ('other', '其他'),
]


class AnnouncementIndexPage(Page):
    """公告列表页"""
    intro = RichTextField(blank=True, verbose_name='介绍')

    content_panels = Page.content_panels + [
        FieldPanel('intro'),
    ]

    class Meta:
        verbose_name = '公告列表页'

    def get_context(self, request):
        context = super().get_context(request)
        
        # 传递地区选项给模板
        context['region_choices'] = REGION_CHOICES
        
        # 获取所有公告
        announcements = AnnouncementPage.objects.live().child_of(self)
        
        # 筛选逻辑
        # 地区筛选
        region = request.GET.get('region')
        if region and region != 'all':
            announcements = announcements.filter(region=region)
        
        # 公告类型筛选
        announcement_type = request.GET.get('type')
        if announcement_type:
            announcements = announcements.filter(announcement_type=announcement_type)
        

        
        # 采购品目筛选
        procurement = request.GET.get('procurement')
        if procurement:
            announcements = announcements.filter(procurement_category=procurement)
        
        # 行业筛选
        industry = request.GET.get('industry')
        if industry:
            announcements = announcements.filter(industry=industry)
        
        # 附件状态筛选
        has_attachment = request.GET.get('has_attachment')
        if has_attachment == 'yes':
            announcements = announcements.exclude(attachment='')
        elif has_attachment == 'no':
            announcements = announcements.filter(attachment='')
        
        # 时间筛选
        from datetime import date, timedelta
        time_filter = request.GET.get('time')
        if time_filter:
            today = date.today()
            if time_filter == 'today':
                announcements = announcements.filter(date=today)
            elif time_filter == 'month':
                announcements = announcements.filter(date__gte=today - timedelta(days=30))
            elif time_filter == 'three_months':
                announcements = announcements.filter(date__gte=today - timedelta(days=90))
            elif time_filter == 'six_months':
                announcements = announcements.filter(date__gte=today - timedelta(days=180))
            elif time_filter == 'year':
                announcements = announcements.filter(date__gte=today - timedelta(days=365))
        
        # 金额筛选（基于采购金额，单位：元）
        amount_range = request.GET.get('amount')
        if amount_range:
            if amount_range == 'below_1':
                # 1万以下
                announcements = announcements.filter(purchase_amount_value__lt=10000)
            elif amount_range == '1_10':
                # 1万到10万
                announcements = announcements.filter(purchase_amount_value__gte=10000, purchase_amount_value__lt=100000)
            elif amount_range == '10_100':
                # 10万到100万
                announcements = announcements.filter(purchase_amount_value__gte=100000, purchase_amount_value__lt=1000000)
            elif amount_range == '100_500':
                # 100万到500万
                announcements = announcements.filter(purchase_amount_value__gte=1000000, purchase_amount_value__lt=5000000)
            elif amount_range == 'above_500':
                # 500万以上
                announcements = announcements.filter(purchase_amount_value__gte=5000000)
        
        # 搜索
        search_query = request.GET.get('search')
        if search_query:
            from django.db.models import Q
            announcements = announcements.filter(Q(title__icontains=search_query) | Q(content__icontains=search_query))
        
        # 排序
        announcements = announcements.order_by('-date')
        
        # 分页
        paginator = Paginator(announcements, 20)  # 每页20条
        page_number = request.GET.get('page', 1)
        page_obj = paginator.get_page(page_number)
        
        context['announcements'] = page_obj
        context['total_count'] = announcements.count()
        
        return context


class AnnouncementPage(Page):
    """公告详情页"""
    date = models.DateField(
        verbose_name="发布日期",
        default=datetime.date.today
    )
    announcement_type = models.CharField(
        max_length=30,
        choices=ANNOUNCEMENT_TYPE_CHOICES,
        default='public_bidding',
        verbose_name='公告类型'
    )
    region = models.CharField(
        max_length=20,
        choices=REGION_CHOICES,
        default='all',
        verbose_name='地区'
    )
    city = models.CharField(
        max_length=100,
        blank=True,
        verbose_name='城市'
    )

    procurement_category = models.CharField(
        max_length=20,
        choices=PROCUREMENT_CATEGORY_CHOICES,
        blank=True,
        verbose_name='采购品目'
    )
    industry = models.CharField(
        max_length=20,
        choices=INDUSTRY_CHOICES,
        default='other',
        verbose_name='所属行业'
    )
    project_number = models.CharField(
        max_length=100,
        blank=True,
        verbose_name='项目编号'
    )
    publisher = models.CharField(
        max_length=200,
        blank=True,
        verbose_name='采购单位'
    )
    agency = models.CharField(
        max_length=200,
        blank=True,
        verbose_name='代理机构'
    )
    purchase_amount_value = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True,
        verbose_name='采购金额（元）'
    )

    @property
    def purchase_amount_wan(self):
        if self.purchase_amount_value:
            return self.purchase_amount_value / 10000
        return None
    content = RichTextField(verbose_name='公告内容')
    attachment = models.FileField(
        upload_to='announcements/attachments/',
        blank=True,
        verbose_name='附件'
    )
    external_link = models.URLField(
        max_length=500,
        blank=True,
        verbose_name='公告原文链接'
    )
    
    # 搜索配置
    search_fields = Page.search_fields + [
        index.SearchField('content'),
        index.SearchField('publisher'),
        index.SearchField('agency'),
        index.FilterField('announcement_type'),
        index.FilterField('region'),
        index.FilterField('industry'),
        index.FilterField('date'),
    ]

    content_panels = Page.content_panels + [
        FieldPanel('date'),
        FieldPanel('announcement_type'),
        FieldPanel('region'),
        FieldPanel('city'),
        FieldPanel('procurement_category'),
        FieldPanel('industry'),
        FieldPanel('project_number'),
        FieldPanel('publisher'),
        FieldPanel('agency'),
        FieldPanel('purchase_amount_value'),
        FieldPanel('content'),
        FieldPanel('attachment'),
        FieldPanel('external_link'),
    ]

    class Meta:
        verbose_name = '公告'

    parent_page_types = ['announcements.AnnouncementIndexPage']


class FavoriteAnnouncement(models.Model):
    """收藏的公告"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='用户')
    announcement = models.ForeignKey(AnnouncementPage, on_delete=models.CASCADE, verbose_name='公告')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='收藏时间')

    class Meta:
        verbose_name = '收藏的公告'
        verbose_name_plural = '收藏的公告'
        unique_together = ['user', 'announcement']
