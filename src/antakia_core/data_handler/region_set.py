import pandas as pd

from antakia_core.data_handler.region import Region, ModelRegion
from antakia_core.data_handler.rules import RuleSet
from antakia_core.utils import boolean_mask, BASE_COLOR


class RegionSet:
    """
    group of regions
    """

    def __init__(self, X):
        """

        Parameters
        ----------
        X: reference dataset
        """
        self.regions = {}
        self.insert_order = []
        self.display_order = []
        self.X = X
        self.left_out_region = Region(self.X,
                                      None,
                                      boolean_mask(self.X, True),
                                      BASE_COLOR,
                                      num=Region.LEFT_OUT_NUM)

    def get_new_num(self) -> int:
        """
        get a new Region id
        Returns
        -------

        """
        if len(self.regions) == 0:
            return 1
        else:
            for i in range(1, len(self.regions) + 1):
                if self.regions.get(i) is None:
                    return i
            return len(self.regions) + 1

    def get_max_num(self) -> int:
        """
        get biggest region id
        Returns
        -------

        """
        if not len(self.regions):
            return 0
        return max(self.insert_order)

    def add(self, region: Region) -> None:
        """
        add a new Region to the set
        prefer the add region method
        Parameters
        ----------
        region

        Returns
        -------

        """
        if region.num < 0:
            num = self.get_new_num()
            region.num = num
        if region.num in self.regions:
            self.remove(region.num)

        self.regions[region.num] = region
        self.insert_order.append(region.num)
        self.display_order.append(region)
        self._compute_left_out_region()

    def add_region(self,
                   rules: RuleSet | None = None,
                   mask=None,
                   color=None,
                   auto_cluster=False) -> Region:
        """
        create a Region from a rule set or a mask
        Parameters
        ----------
        rules : rule list
        mask : selection mask
        color : region color
        auto_cluster: is from autoclustering ?

        Returns
        -------
        the created region

        """
        if mask is not None:
            mask = mask.reindex(self.X.index).fillna(False)
        region = Region(X=self.X, rules=rules, mask=mask, color=color)
        region.auto_cluster = auto_cluster
        self.add(region)
        return region

    def extend(self, region_set: 'RegionSet') -> None:
        """
        add the provided RegionSet into the current one
        rebuilds all Regions
        Parameters
        ----------
        region_set

        Returns
        -------

        """
        for region in region_set.regions.values():
            self.add_region(region.rules, region.mask, region._color,
                            region.auto_cluster)

    def remove(self, region_num) -> None:
        """
        remove Region from set
        Parameters
        ----------
        region_num

        Returns
        -------

        """
        if region_num != Region.LEFT_OUT_NUM:
            self.insert_order.remove(region_num)
            self.display_order.remove(self.regions[region_num])
            del self.regions[region_num]
        self._compute_left_out_region()

    def to_dict(self, include_left_out=True) -> list[dict]:
        """
        dict like RegionSet
        Returns
        -------

        """
        region_dicts = [region.to_dict() for region in self.display_order]
        if include_left_out:
            region_dicts += [self.left_out_region.to_dict()]
        return region_dicts

    def get_masks(self) -> list[pd.Series]:
        """
        returns all Region masks
        Returns
        -------

        """
        return [region.mask for region in self.display_order]

    @property
    def mask(self):
        """
        get the union mask of all regions
        Returns
        -------

        """
        union_mask = boolean_mask(self.X, False)
        for mask in self.get_masks():
            union_mask |= mask
        return union_mask

    def get_colors(self) -> list[str]:
        """
        get the list of Region colors
        Returns
        -------

        """
        return [region.color for region in self.display_order]

    def get_color_serie(self) -> pd.Series:
        """
        get a pd.Series with for each sample of self.X its region color
        the value is set to grey if the sample is not in any Region of the region set
        Returns
        -------

        """
        color = pd.Series([BASE_COLOR] * len(self.X), index=self.X.index)
        for region in self.display_order:
            color[region.mask] = region.color
        return color

    def __len__(self) -> int:
        """
        size of the region set
        Returns
        -------

        """
        return len(self.regions)

    def get(self, i) -> Region | None:
        """
        get a specific region by id
        Parameters
        ----------
        i

        Returns
        -------

        """
        if i == Region.LEFT_OUT_NUM:
            return self.left_out_region
        return self.regions.get(i)

    def clear_unvalidated(self):
        """
        remove all unvalidated regions
        Returns
        -------

        """
        for i in list(self.regions.keys()):
            if not self.regions[i].validated:
                self.remove(i)

    def pop_last(self) -> Region | None:
        """
        removes and return the last region
        Returns
        -------

        """
        if len(self.insert_order) > 0:
            num = self.insert_order[-1]
            region = self.get(num)
            if not self.regions[num].validated:
                self.remove(num)
            return region
        return None

    def sort(self, by, ascending=True):
        """
        sort the region set by id, size, insert order
        Parameters
        ----------
        by : 'region_num'|'size'|'insert'
        ascending

        Returns
        -------

        """
        if by == 'region_num':
            key = lambda x: x.num
        elif by == 'size':
            key = lambda x: x.num_points()
        elif by == 'insert':
            key = lambda x: self.insert_order.index(x)
        self.display_order.sort(key=key, reverse=not ascending)

    def stats(self) -> dict:
        """ Computes the number of distinct points in the regions and the coverage in %
        """
        union_mask = self.mask
        stats = {
            'regions': len(self),
            'points': union_mask.sum(),
            'coverage': round(100 * union_mask.mean()),
        }
        return stats

    def _compute_left_out_region(self):
        """
        compute the left out region
        Returns
        -------

        """
        left_out_mask = ~self.mask
        self.left_out_region.update_mask(left_out_mask)
        return self.left_out_region


class ModelRegionSet(RegionSet):
    """
    Supercharged RegionSet to handle interpretable models
    """

    def __init__(self, X, y, X_test, y_test, model, score):
        """

        Parameters
        ----------
        X: reference DatafFrame
        y: target series
        X_test: test set
        y_test: target test set
        model: customer model
        score: scoring method
        """
        super().__init__(X)
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.score = score
        self.left_out_region = self.upgrade_region_to_model_region(
            self.left_out_region)

    def upgrade_region_to_model_region(self, region: Region):
        """
        Upgrade the provided region to a model region
        Parameters
        ----------
        region

        Returns
        -------

        """
        model_region = ModelRegion(X=self.X,
                                   y=self.y,
                                   X_test=self.X_test,
                                   y_test=self.y_test,
                                   customer_model=self.model,
                                   score=self.score,
                                   rules=region.rules,
                                   mask=region.mask,
                                   color=region._color,
                                   num=region.num)
        model_region.validated = region.validated
        return model_region

    def add(self, region: Region):
        if not isinstance(region, ModelRegion):
            region = self.upgrade_region_to_model_region(region)
        super().add(region)

    def add_region(self,
                   rules: RuleSet | None = None,
                   mask=None,
                   color=None,
                   auto_cluster=False) -> Region:
        """
        add new ModelRegion
        Parameters
        ----------
        rules
        mask
        color
        auto_cluster

        Returns
        -------

        """
        if mask is not None:
            mask = mask.reindex(self.X.index).fillna(False)
        region = ModelRegion(X=self.X,
                             y=self.y,
                             X_test=self.X_test,
                             y_test=self.y_test,
                             customer_model=self.model,
                             score=self.score,
                             rules=rules,
                             mask=mask,
                             color=color)
        region.num = -1
        region.auto_cluster = auto_cluster
        self.add(region)
        return region

    def get(self, i) -> ModelRegion | None:
        return super().get(i)  # type:ignore

    def stats(self) -> dict:
        base_stats = super().stats()
        delta_score = 0.
        for region in self.regions.values():
            weight = region.mask.sum()
            delta = region.delta
            delta_score += weight * delta
        delta_score /= len(self.X)
        base_stats['delta_score'] = delta_score
        return base_stats

    def predict(self, X):
        prediction = pd.Series(index=X.index)
        for region in self.regions.values():
            prediction = prediction.fillna(region.predict(X))
        return prediction
