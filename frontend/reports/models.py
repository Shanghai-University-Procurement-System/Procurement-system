from django.db import models
from django import forms
from wagtail.models import Page, Orderable
from wagtail.fields import RichTextField
from wagtail.admin.panels import FieldPanel, InlinePanel
from wagtail.search import index
from django.core.paginator import Paginator
from modelcluster.fields import ParentalKey
from django.utils import timezone
import os
import datetime


# 报告分类选择
REPORT_CATEGORY_CHOICES = [
    ('default', '默认'),
    ('analysis', '调查分析'),
    ('ai', '人工智能'),
    ('bigdata', '大数据'),
    ('scope', '中成氏范围'),
    ('opinion', '舆论研究'),
    ('iot', '物联'),
    ('life', '生活'),
]


class ReportIndexPage(Page):
    """需求调查报告列表页"""
    intro = RichTextField(blank=True, verbose_name='介绍')

    content_panels = Page.content_panels + [
        FieldPanel('intro'),
    ]

    class Meta:
        verbose_name = '需求调查报告列表页'

    def get_context(self, request):
        context = super().get_context(request)
        
        # 获取所有报告
        reports = ReportPage.objects.live().child_of(self)

        # Filter by owner (User-specific visibility)
        if request.user.is_authenticated:
            if request.user.is_superuser:
                pass  # 超级管理员看到所有报告
            else:
                # 普通用户看到：自己的报告 + 公共报告
                from django.db.models import Q
                reports = reports.filter(
                    Q(owner=request.user) | Q(is_public=True)
                )
        else:
            # 未登录用户看不到报告
            reports = ReportPage.objects.none()
        
        # 分类筛选
        category = request.GET.get('category')
        if category and category != 'default':
            reports = reports.filter(category=category)
        
        # 搜索
        search_query = request.GET.get('search')
        if search_query:
            reports = reports.search(search_query)
        else:
            # 仅在非搜索模式下排序（搜索结果按相关性排列）
            reports = reports.order_by('-analysis_time')
        
        # 分页
        paginator = Paginator(reports, 20)  # 每页20条
        page_number = request.GET.get('page', 1)
        page_obj = paginator.get_page(page_number)
        
        context['reports'] = page_obj
        try:
            context['total_count'] = reports.count()
        except (AttributeError, TypeError):
            context['total_count'] = len(reports)
        
        return context


class HistoricalProject(Orderable):
    page = ParentalKey('ReportPage', on_delete=models.CASCADE, related_name='historical_projects')
    project_name = models.CharField(max_length=255, verbose_name='项目名称', blank=True, null=True)
    procurement_method = models.CharField(max_length=100, verbose_name='采购方式', blank=True, null=True)
    procurement_time = models.DateField(verbose_name='采购时间', blank=True, null=True)
    procurement_unit = models.CharField(max_length=255, verbose_name='采购单位', blank=True, null=True)
    budget_amount = models.DecimalField(max_digits=15, decimal_places=2, verbose_name='预算(万元)', blank=True, null=True)
    supplier_name = models.CharField(max_length=255, verbose_name='供应商名称', blank=True, null=True)
    transaction_amount = models.DecimalField(max_digits=15, decimal_places=2, verbose_name='成交(万元)', blank=True, null=True)
    detail_link = models.URLField(verbose_name='详情链接', blank=True, null=True)

    panels = [
        FieldPanel('project_name'),
        FieldPanel('procurement_method'),
        FieldPanel('procurement_time'),
        FieldPanel('procurement_unit'),
        FieldPanel('budget_amount'),
        FieldPanel('supplier_name'),
        FieldPanel('transaction_amount'),
        FieldPanel('supplier_name'),
        FieldPanel('transaction_amount'),
        FieldPanel('detail_link'),
    ]

    panels = [
        FieldPanel('project_name'),
        FieldPanel('procurement_method'),
        FieldPanel('procurement_time'),
        FieldPanel('procurement_unit'),
        FieldPanel('budget_amount'),
        FieldPanel('supplier_name'),
        FieldPanel('transaction_amount'),
        FieldPanel('detail_link'),
    ]


class ReportAnnouncement(Orderable):
    page = ParentalKey('ReportPage', on_delete=models.CASCADE, related_name='report_announcements')
    title = models.CharField(max_length=255, verbose_name='公告标题')
    url = models.URLField(verbose_name='公告链接', blank=True, null=True)
    publish_date = models.DateField(verbose_name='发布时间', blank=True, null=True)
    
    ANNOUNCEMENT_TYPES = [
        ('bidding', '招标公告'),
        ('result', '结果公告'),
        ('change', '变更公告'),
        ('termination', '终止/废标公告'),
        ('contract', '合同公示'),
        ('acceptance', '验收公示'),
        ('other', '其他公告'),
    ]
    announcement_type = models.CharField(max_length=50, choices=ANNOUNCEMENT_TYPES, default='bidding', verbose_name='公告类型')
    
    # Simple region field or choices. Using CharField for flexibility as per plan, but could use choices if needed to match filter.
    # Matching existing choices from other models is safer.
    PROVINCE_CHOICES = [
        ('全国', '全国'), ('北京', '北京'), ('天津', '天津'), ('河北', '河北'), 
        ('山西', '山西'), ('内蒙古', '内蒙古'), ('辽宁', '辽宁'), ('吉林', '吉林'), 
        ('黑龙江', '黑龙江'), ('上海', '上海'), ('江苏', '江苏'), ('浙江', '浙江'), 
        ('安徽', '安徽'), ('福建', '福建'), ('江西', '江西'), ('山东', '山东'), 
        ('河南', '河南'), ('湖北', '湖北'), ('湖南', '湖南'), ('广东', '广东'), 
        ('广西', '广西'), ('海南', '海南'), ('重庆', '重庆'), ('四川', '四川'), 
        ('贵州', '贵州'), ('云南', '云南'), ('西藏', '西藏'), ('陕西', '陕西'), 
        ('甘肃', '甘肃'), ('青海', '青海'), ('宁夏', '宁夏'), ('新疆', '新疆'), 
        ('香港', '香港'), ('澳门', '澳门'), ('台湾', '台湾')
    ]
    region = models.CharField(max_length=50, choices=PROVINCE_CHOICES, default='全国', verbose_name='地区')
    
    budget_amount = models.DecimalField(max_digits=15, decimal_places=2, verbose_name='预算金额(万元)', blank=True, null=True)
    transaction_amount = models.DecimalField(max_digits=15, decimal_places=2, verbose_name='成交金额(万元)', blank=True, null=True)
    buyer = models.CharField(max_length=255, verbose_name='采购单位', blank=True, null=True)
    project_number = models.CharField(max_length=100, verbose_name='项目编号', blank=True, null=True)

    panels = [
        FieldPanel('title'),
        FieldPanel('url'),
        FieldPanel('publish_date'),
        FieldPanel('announcement_type'),
        FieldPanel('region'),
        FieldPanel('budget_amount'),
        FieldPanel('transaction_amount'),
        FieldPanel('buyer'),
        FieldPanel('project_number'),
    ]



class ReportContract(Orderable):
    page = ParentalKey('ReportPage', on_delete=models.CASCADE, related_name='report_contracts')
    title = models.CharField(max_length=255, verbose_name='合同名称')
    url = models.URLField(verbose_name='详情链接', blank=True, null=True)
    publish_date = models.DateField(verbose_name='发布时间', blank=True, null=True)
    
    PROVINCE_CHOICES = [
        ('全国', '全国'), ('北京', '北京'), ('天津', '天津'), ('河北', '河北'), 
        ('山西', '山西'), ('内蒙古', '内蒙古'), ('辽宁', '辽宁'), ('吉林', '吉林'), 
        ('黑龙江', '黑龙江'), ('上海', '上海'), ('江苏', '江苏'), ('浙江', '浙江'), 
        ('安徽', '安徽'), ('福建', '福建'), ('江西', '江西'), ('山东', '山东'), 
        ('河南', '河南'), ('湖北', '湖北'), ('湖南', '湖南'), ('广东', '广东'), 
        ('广西', '广西'), ('海南', '海南'), ('重庆', '重庆'), ('四川', '四川'), 
        ('贵州', '贵州'), ('云南', '云南'), ('西藏', '西藏'), ('陕西', '陕西'), 
        ('甘肃', '甘肃'), ('青海', '青海'), ('宁夏', '宁夏'), ('新疆', '新疆'), 
        ('香港', '香港'), ('澳门', '澳门'), ('台湾', '台湾')
    ]
    region = models.CharField(max_length=50, choices=PROVINCE_CHOICES, default='全国', verbose_name='地区')
    city = models.CharField(max_length=50, verbose_name='城市', blank=True, null=True)
    
    budget_amount = models.DecimalField(max_digits=15, decimal_places=2, verbose_name='预算金额(万元)', blank=True, null=True)
    transaction_amount = models.DecimalField(max_digits=15, decimal_places=2, verbose_name='成交金额(万元)', blank=True, null=True)
    buyer = models.CharField(max_length=255, verbose_name='采购单位', blank=True, null=True)

    panels = [
        FieldPanel('title'),
        FieldPanel('url'),
        FieldPanel('publish_date'),
        FieldPanel('region'),
        FieldPanel('city'),
        FieldPanel('budget_amount'),
        FieldPanel('transaction_amount'),
        FieldPanel('buyer'),
    ]


class ReportBiddingDocument(Orderable):
    page = ParentalKey('ReportPage', on_delete=models.CASCADE, related_name='report_documents')
    title = models.CharField(max_length=255, verbose_name='文件标题', blank=True, default='')
    source = models.CharField(max_length=100, verbose_name='来源', default='山东政府采购网', blank=True)
    upload_time = models.DateField(verbose_name='上传时间', default=datetime.date.today)
    file = models.FileField(upload_to='reports/documents/', verbose_name='文件', blank=True)
    
    DOC_TYPES = [
        ('procurement', '采购文件'),
        ('contract', '合同文件'),
        ('acceptance', '验收文件'),
        ('other', '其他文件'),
    ]
    doc_type = models.CharField(max_length=50, choices=DOC_TYPES, default='procurement', verbose_name='文件类型')

    def clean(self):
        print(f"DEBUG: ReportBiddingDocument clean() called. Title: {self.title}, File: {self.file}")
        super().clean()

    def save(self, *args, **kwargs):
        # Auto-fill title from filename if empty
        if not self.title and self.file:
            try:
                # Get the filename
                filename = os.path.basename(self.file.name)
                # Ensure it's not too long for the field
                if len(filename) > 255:
                    filename = filename[:255]
                self.title = filename
            except Exception:
                pass
        super().save(*args, **kwargs)

    @property
    def filename(self):
        import os
        return os.path.basename(self.file.name)

    panels = [
        FieldPanel('title'),
        FieldPanel('source'),
        FieldPanel('upload_time'),
        FieldPanel('file'),
        FieldPanel('doc_type'),
    ]


class OngoingProject(Orderable):
    page = ParentalKey('ReportPage', on_delete=models.CASCADE, related_name='ongoing_projects')
    project_name = models.CharField(max_length=255, verbose_name='项目名称', blank=True, null=True)
    bid_opening_time = models.DateField(verbose_name='开标时间', blank=True, null=True)
    procurement_unit = models.CharField(max_length=255, verbose_name='采购单位', blank=True, null=True)
    budget_amount = models.DecimalField(max_digits=15, decimal_places=2, verbose_name='预算(万元)', blank=True, null=True)
    detail_link = models.URLField(verbose_name='详情链接', blank=True, null=True)

    panels = [
        FieldPanel('project_name'),
        FieldPanel('bid_opening_time'),
        FieldPanel('procurement_unit'),
        FieldPanel('budget_amount'),
        FieldPanel('detail_link'),
    ]


class PurchaseIntention(Orderable):
    page = ParentalKey('ReportPage', on_delete=models.CASCADE, related_name='purchase_intentions')
    project_name = models.CharField(max_length=255, verbose_name='项目名称')
    budget_amount = models.DecimalField(max_digits=15, decimal_places=2, verbose_name='预算(万元)', blank=True, null=True)
    
    PROVINCE_CHOICES = [
        ('全国', '全国'), ('北京', '北京'), ('天津', '天津'), ('河北', '河北'), 
        ('山西', '山西'), ('内蒙古', '内蒙古'), ('辽宁', '辽宁'), ('吉林', '吉林'), 
        ('黑龙江', '黑龙江'), ('上海', '上海'), ('江苏', '江苏'), ('浙江', '浙江'), 
        ('安徽', '安徽'), ('福建', '福建'), ('江西', '江西'), ('山东', '山东'), 
        ('河南', '河南'), ('湖北', '湖北'), ('湖南', '湖南'), ('广东', '广东'), 
        ('广西', '广西'), ('海南', '海南'), ('重庆', '重庆'), ('四川', '四川'), 
        ('贵州', '贵州'), ('云南', '云南'), ('西藏', '西藏'), ('陕西', '陕西'), 
        ('甘肃', '甘肃'), ('青海', '青海'), ('宁夏', '宁夏'), ('新疆', '新疆'), 
        ('香港', '香港'), ('澳门', '澳门'), ('台湾', '台湾')
    ]
    province = models.CharField(max_length=50, choices=PROVINCE_CHOICES, default='全国', verbose_name='地区（省份）')
    city = models.CharField(max_length=50, verbose_name='城市', blank=True, null=True)
    
    procurement_category = models.CharField(max_length=100, verbose_name='采购品目', blank=True, null=True)
    procurement_unit = models.CharField(max_length=255, verbose_name='采购单位', blank=True, null=True)
    publish_time = models.DateField(verbose_name='发布时间', blank=True, null=True)
    content = models.TextField(verbose_name='采购内容', blank=True, null=True)
    detail_url = models.URLField(verbose_name='详情链接', blank=True, null=True)

    panels = [
        FieldPanel('project_name'),
        FieldPanel('budget_amount'),
        FieldPanel('province'),
        FieldPanel('city'),
        FieldPanel('procurement_category'),
        FieldPanel('procurement_unit'),
        FieldPanel('publish_time'),
        FieldPanel('content'),
        FieldPanel('detail_url'),
    ]


class TopBuyerAnalysis(Orderable):
    """各单位采购情况分析 (Top 5)"""
    page = ParentalKey('ReportPage', related_name='top_buyers')
    buyer_name = models.CharField(max_length=255, verbose_name="采购单位")
    region = models.CharField(max_length=100, verbose_name="所属地区", blank=True, default="-")
    project_count = models.IntegerField(verbose_name="项目数(项)", default=0)
    budget_amount = models.CharField(max_length=50, verbose_name="预算金额(万元)", blank=True, default="-")
    transaction_amount = models.CharField(max_length=50, verbose_name="成交金额(万元)", blank=True, default="-")

    panels = [
        FieldPanel('buyer_name'),
        FieldPanel('region'),
        FieldPanel('project_count'),
        FieldPanel('budget_amount'),
        FieldPanel('transaction_amount'),
    ]


class RegionPurchaseData(Orderable):
    """各地区采购项目数据（用于图表展示）"""
    page = ParentalKey('ReportPage', related_name='region_purchase_data')
    region_name = models.CharField(max_length=50, verbose_name="地区名称", help_text="例如：浙江、湖南、四川")
    purchase_amount = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="采购金额(万元)", default=0)
    project_count = models.IntegerField(verbose_name="项目数量(个)", default=0)

    panels = [
        FieldPanel('region_name'),
        FieldPanel('purchase_amount'),
        FieldPanel('project_count'),
    ]

    class Meta(Orderable.Meta):
        verbose_name = "地区采购数据"


class TopSupplierShare(Orderable):
    """成交前5名供应商市场份额占比"""
    page = ParentalKey('ReportPage', related_name='top_suppliers')
    supplier_name = models.CharField(max_length=255, verbose_name="供应商名称")
    project_count_percent = models.DecimalField(max_digits=6, decimal_places=2, verbose_name="项目数量占比(%)", default=0)
    transaction_amount_percent = models.DecimalField(max_digits=6, decimal_places=2, verbose_name="成交金额占比(%)", default=0)

    panels = [
        FieldPanel('supplier_name'),
        FieldPanel('project_count_percent'),
        FieldPanel('transaction_amount_percent'),
    ]

    class Meta(Orderable.Meta):
        verbose_name = "供应商市场份额"


class ProcurementMethodShare(Orderable):
    """各采购方式项目数量占比"""
    page = ParentalKey('ReportPage', related_name='procurement_methods')
    method_name = models.CharField(max_length=100, verbose_name="采购方式", help_text="例如：公开招标、竞争性磋商、单一来源")
    project_count_percent = models.DecimalField(max_digits=6, decimal_places=2, verbose_name="项目数量占比(%)", default=0)

    panels = [
        FieldPanel('method_name'),
        FieldPanel('project_count_percent'),
    ]

    class Meta(Orderable.Meta):
        verbose_name = "采购方式占比"


class ReportPage(Page):
    template = "reports/report_page_v3.html"
    
    # Reports fields
    """需求调查报告详情页"""

    """需求调查报告详情页"""
    serial_number = models.CharField(max_length=50, blank=True, null=True, verbose_name='序号')
    procurement_name = models.TextField(verbose_name='拟采购名称')
    
    ANALYSIS_CHOICES = [
        ('一键分析', '一键分析'),
        ('高级分析', '高级分析'),
    ]
    analysis_type = models.CharField(
        max_length=50,
        choices=ANALYSIS_CHOICES,
        default='一键分析',
        verbose_name='选择制度'
    )
    analysis_time = models.DateTimeField(auto_now_add=True, verbose_name='分析时间')
    category = models.CharField(
        max_length=20,
        choices=REPORT_CATEGORY_CHOICES,
        default='default',
        verbose_name='分类'
    )
    
    # New analysis text fields
    market_supply_analysis = models.TextField(blank=True, verbose_name='市场供给分析')
    ai_summary_analysis = models.TextField(blank=True, verbose_name='AI总结分析')

    # 市场供给分析 - 统计卡片字段
    stat_buyer_count = models.CharField(max_length=50, blank=True, default='', verbose_name='采购单位数量', help_text='例如：268 家')
    stat_region_count = models.CharField(max_length=50, blank=True, default='', verbose_name='采购地区数量', help_text='例如：31 个地区')
    stat_budget_total = models.CharField(max_length=50, blank=True, default='', verbose_name='预算金额', help_text='例如：153,674 万元')
    stat_transaction_total = models.CharField(max_length=50, blank=True, default='', verbose_name='成交总额', help_text='例如：125,809 万元')
    stat_announcement_count = models.CharField(max_length=50, blank=True, default='', verbose_name='采购公告数量', help_text='例如：4235 条')

    summary = RichTextField(blank=True, verbose_name='报告摘要')
    content = RichTextField(blank=True, verbose_name='报告内容')
    is_public = models.BooleanField(
        default=False,
        verbose_name='公共报告',
        help_text='勾选后所有用户都能看到此报告（管理员修改后所有人同步更新）'
    )

    pdf_file = models.FileField(
        upload_to='reports/pdfs/',
        blank=True,
        verbose_name='PDF文件'
    )
    
    # 搜索配置
    search_fields = Page.search_fields + [
        index.SearchField('procurement_name'),
        index.SearchField('summary'),
        index.SearchField('content'),
        index.FilterField('category'),
    ]

    content_panels = Page.content_panels + [
        FieldPanel('is_public'),
        FieldPanel('serial_number'),
        FieldPanel('procurement_name'),
        FieldPanel('analysis_type'),
        FieldPanel('pdf_file'),
        FieldPanel('content'),
        InlinePanel('report_announcements', label="采购公告"),
        InlinePanel('report_contracts', label="采购合同"),
        InlinePanel('report_documents', label="招标文件"),
        InlinePanel('purchase_intentions', label="采购意向"),
        InlinePanel('historical_projects', label="历史成交项目"),
        InlinePanel('ongoing_projects', label="进行中项目"),
        InlinePanel('top_buyers', label="各单位采购情况分析 (前5名)"),
        InlinePanel('region_purchase_data', label="各地区采购数据 (图表)"),
        InlinePanel('top_suppliers', label="供应商市场份额占比 (前5名)"),
        InlinePanel('procurement_methods', label="采购方式占比"),
        FieldPanel('market_supply_analysis'),
        FieldPanel('ai_summary_analysis'),
        FieldPanel('stat_buyer_count'),
        FieldPanel('stat_region_count'),
        FieldPanel('stat_budget_total'),
        FieldPanel('stat_transaction_total'),
        FieldPanel('stat_announcement_count'),
    ]


    class Meta:
        verbose_name = '需求调查报告'

    parent_page_types = ['reports.ReportIndexPage']
